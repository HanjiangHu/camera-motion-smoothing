# Code modified based on  https://github.com/AI-secure/semantic-randomized-smoothing

import os
import sys


import argparse
import torch
import torchvision
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS, get_normalize_layer
from architectures import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from core import SemanticSmooth
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from tensorboardX import SummaryWriter
from train_utils import AverageMeter, accuracy, init_logfile, log
from transformers import gen_transformer, AbstractTransformer
from tqdm import trange

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['resolvable_tz', 'resolvable_tx', 'resolvable_ty', 'resolvable_rz', 'resolvable_rx',
                             'resolvable_ry'])
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--rotation_angle', help='constrain the rotation angle to +-rotation angle in degree',
                    type=float, default=180.0)
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument('--blur_lamb', default=0.0, type=float,
                    help="standard deviation of Exponential Gaussian blur, only useful when transtype is universal")
parser.add_argument('--sigma_trans', default=0.0, type=float,
                    help="standard deviation of translation, only useful when transtype is universal")
parser.add_argument('--sl', default=1.0, type=float,
                    help="resize minimum ratio")
parser.add_argument('--sr', default=1.0, type=float,
                    help="resize maximum ratio")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrain', default=None, type=str)
##################### arguments for consistency training #####################
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=20., type=float)
##################### arguments for tensorboard print #####################
parser.add_argument('--print_step', action="store_true")
parser.add_argument('--vanilla', action="store_true")
parser.add_argument('--uniform', action="store_true")
parser.add_argument('--beta', action="store_true")
parser.add_argument('--benign', action="store_true")
parser.add_argument('--proj_test', action="store_true")
parser.add_argument('--sample_num', default=100, type=int)
parser.add_argument('--base', action="store_true")
args = parser.parse_args()


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets,
                    reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.proj_test:
        test_dataset = get_dataset(args.dataset, 'proj_test', args.transtype, args.vanilla, args.uniform, args.beta)
    else:
        # empirical test and certify
        test_dataset = get_dataset(args.dataset, 'certify', args.transtype, args.vanilla, args.uniform, args.beta)
    pin_memory = (args.dataset == "imagenet") or (args.dataset == "metaroom")

    model = get_architecture(args.arch, args.dataset)

    if args.pretrain is not None:
        if args.pretrain == 'torchvision':
            # load pretrain model from torchvision
            if args.dataset == 'imagenet' or args.dataset == 'metaroom':# and args.arch == 'resnet50':
                model = torchvision.models.resnet50(True).cuda()

                # fix
                normalize_layer = get_normalize_layer(args.dataset).cuda()
                model = torch.nn.Sequential(normalize_layer, model)


                print('loaded from torchvision for imagenet resnet50')
            else:
                raise Exception(f'Unsupported pretrain arg {args.pretrain}')
        else:
            # load the base classifier
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded from {args.pretrain}')

    if args.noise_sd == 0.0:
        logfilename = os.path.join(args.outdir, 'empirical_test_log.txt')
    else:
        logfilename = os.path.join(args.outdir, f'rand_empirical_{args.noise_sd}_test_log.txt')
    # init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    writer = SummaryWriter(args.outdir)


    transformer = gen_transformer(args, test_dataset[0][0])

    criterion = CrossEntropyLoss().cuda()

    before = time.time()
    train_loss = -1.0
    train_acc = -1.0
    test_loss, test_acc = test(test_dataset, model, criterion, 0, transformer, writer, print_freq=args.print_freq)
    after = time.time()


    log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
        0, str(datetime.timedelta(seconds=(after - before))),
        -1.0+0.01, train_loss+0.01, train_acc+0.01, test_loss+0.01, test_acc))


def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]






def test(dataset, model, criterion, epoch, transformer: AbstractTransformer, writer=True, print_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    if args.uniform:
        type = "uniform"
    elif args.beta:
        type = "beta"
    else:
        type = "benign"
    smoothed_classifier = SemanticSmooth(model, get_num_classes(args.dataset), transformer)
    with torch.no_grad():
        if args.base:
            for i in range(len(dataset)):
                (inputs_raw, targets_raw) = dataset[i]
                # for i, (inputs, targets) in enumerate(loader):
                # measure data loading time
                data_time.update(time.time() - end)
                correct_100 = True

                # print(i, targets)
                inputs_raw = [inputs_raw]
                targets = torch.tensor(targets_raw).unsqueeze(0).cuda()
                # print(targets.shape, targets)
                # print(inputs_raw)
                assert len(inputs_raw) == 1
                for _ in range(args.sample_num):
                    # augment inputs with noise
                    # print(transformer.process(inputs, empirical=True, type=type)[0].shape)

                    # compute output

                    inputs = transformer.process(inputs_raw, empirical=True, type=type)[0].unsqueeze(0).cuda()
                    inputs = torch.transpose(inputs, 2, 3)
                    inputs = torch.transpose(inputs, 1, 2).type(torch.cuda.FloatTensor)
                    outputs = model(inputs)

                    # loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                    if acc1.item() != 100.0:
                        correct_100 = False
                        break
                    if args.benign:
                        break
                if correct_100:
                    losses.update(0.0, inputs.size(0))
                    top1.update(100.0, inputs.size(0))
                    top5.update(0.0, inputs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.avg:.4f}\t'
                              'Acc@1 {top1.avg:.3f}\t'
                              'Acc@5 {top5.avg:.3f}'.format(
                            i, len(dataset), batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1, top5=top5))
                    # correct_100 = True
                else:
                    correct_100 = True

                    losses.update(0.0, inputs.size(0))
                    top1.update(0.0, inputs.size(0))
                    top5.update(0.0, inputs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.avg:.4f}\t'
                              'Acc@1 {top1.avg:.3f}\t'
                              'Acc@5 {top5.avg:.3f}'.format(
                            i, len(dataset), batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1, top5=top5))

            if writer:
                writer.add_scalar('loss/test', losses.avg, epoch)
                writer.add_scalar('accuracy/test@1', top1.avg, epoch)
                writer.add_scalar('accuracy/test@5', top5.avg, epoch)
            loss, top1 = losses.avg, top1.avg
        else:
            tot, tot_good = 0, 0
            for i in range(len(dataset)):
                (inputs_raw, targets_raw) = dataset[i]
                correct_100 = True
                # inputs_raw = [inputs_raw]
                for _ in trange(args.sample_num):
                    inputs = transformer.projection_adder.pertubate(inputs_raw, empirical=True, type=type)
                    outputs = smoothed_classifier.predict(inputs, 100, 0.01, 100)
                    clean_correct = (outputs == targets_raw)

                    if not clean_correct:
                        correct_100 = False
                        break
                    if args.benign:
                        break
                if correct_100:
                    tot, tot_good = tot + 1, tot_good + 1
                else:
                    tot, tot_good = tot + 1, tot_good
                print(f'{i} RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}')
            loss, top1 = 0, float(tot_good) / float(tot)
        return (loss, top1)


if __name__ == "__main__":
    main()

