python empirical_test.py metaroom resnet18 resolvable_tz data/predict/metaroom/resnet18/tz/tv_noise_0.1_uniform5 --batch 1 --noise_sd 0.1  --gpu 0 --pretrain ./models/metaroom/resnet18/tz/tv_noise_0.1/checkpoint.pth.tar --uniform
python empirical_test.py metaroom resnet18 resolvable_tz data/predict/metaroom/resnet18/tz/tv_noise_0.1_benign_sm --batch 1 --noise_sd 0.1  --gpu 0 --pretrain ./models/metaroom/resnet18/tz/tv_noise_0.1/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_tz data/predict/metaroom/resnet18/tz/tv_va_uniform5 --batch 1 --noise_sd 0.1  --gpu 0 --pretrain ./models/metaroom/resnet18/tz/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_tz data/predict/metaroom/resnet18/tz/tv_va_benign_sm --batch 1 --noise_sd 0.1  --gpu 0 --pretrain ./models/metaroom/resnet18/tz/tv_va/checkpoint.pth.tar --benign --vanilla


python empirical_test.py metaroom resnet18 resolvable_tx data/predict/metaroom/resnet18/tx/tv_noise_0.05_uniform5 --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/tx/tv_noise_0.05/checkpoint.pth.tar --uniform 
python empirical_test.py metaroom resnet18 resolvable_tx data/predict/metaroom/resnet18/tx/tv_noise_0.05_benign_sm --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/tx/tv_noise_0.05/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_tx data/predict/metaroom/resnet18/tx/tv_va_uniform5 --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/tx/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_tx data/predict/metaroom/resnet18/tx/tv_va_benign_sm --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/tx/tv_va/checkpoint.pth.tar --benign --vanilla


python empirical_test.py metaroom resnet18 resolvable_ty data/predict/metaroom/resnet18/ty/tv_noise_0.05_uniform5 --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/ty/tv_noise_0.05/checkpoint.pth.tar --uniform 
python empirical_test.py metaroom resnet18 resolvable_ty data/predict/metaroom/resnet18/ty/tv_noise_0.05_benign_sm --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/ty/tv_noise_0.05/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_ty data/predict/metaroom/resnet18/ty/tv_va_uniform5 --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/ty/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_ty data/predict/metaroom/resnet18/ty/tv_va_benign_sm --batch 1 --noise_sd 0.05  --gpu 0 --pretrain ./models/metaroom/resnet18/ty/tv_va/checkpoint.pth.tar --benign --vanilla


python empirical_test.py metaroom resnet18 resolvable_rz data/predict/metaroom/resnet18/rz/tv_noise_7_uniform5 --batch 1 --noise_sd 0.122173  --gpu 0 --pretrain ./models/metaroom/resnet18/rz/tv_noise_7/checkpoint.pth.tar --uniform 
python empirical_test.py metaroom resnet18 resolvable_rz data/predict/metaroom/resnet18/rz/tv_noise_7_benign_sm --batch 1 --noise_sd 0.122173  --gpu 0 --pretrain ./models/metaroom/resnet18/rz/tv_noise_7/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_rz data/predict/metaroom/resnet18/rz/tv_va_uniform5 --batch 1 --noise_sd 0.122173  --gpu 0 --pretrain ./models/metaroom/resnet18/rz/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_rz data/predict/metaroom/resnet18/rz/tv_va_benign_sm --batch 1 --noise_sd 0.122173  --gpu 0 --pretrain ./models/metaroom/resnet18/rz/tv_va/checkpoint.pth.tar --benign --vanilla


python empirical_test.py metaroom resnet18 resolvable_rx data/predict/metaroom/resnet18/rx/tv_noise_2.5_uniform5 --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/rx/tv_noise_2.5/checkpoint.pth.tar --uniform 
python empirical_test.py metaroom resnet18 resolvable_rx data/predict/metaroom/resnet18/rx/tv_noise_2.5_benign_sm --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/rx/tv_noise_2.5/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_rx data/predict/metaroom/resnet18/rx/tv_va_uniform5 --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/rx/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_rx data/predict/metaroom/resnet18/rx/tv_va_benign_sm --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/rx/tv_va/checkpoint.pth.tar --benign --vanilla


python empirical_test.py metaroom resnet18 resolvable_ry data/predict/metaroom/resnet18/ry/tv_noise_2.5_uniform5 --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/ry/tv_noise_2.5/checkpoint.pth.tar --uniform 
python empirical_test.py metaroom resnet18 resolvable_ry data/predict/metaroom/resnet18/ry/tv_noise_2.5_benign_sm --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/ry/tv_noise_2.5/checkpoint.pth.tar --benign
python empirical_test.py metaroom resnet18 resolvable_ry data/predict/metaroom/resnet18/ry/tv_va_uniform5 --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/ry/tv_va/checkpoint.pth.tar --uniform --vanilla
python empirical_test.py metaroom resnet18 resolvable_ry data/predict/metaroom/resnet18/ry/tv_va_benign_sm --batch 1 --noise_sd 0.04363323  --gpu 0 --pretrain ./models/metaroom/resnet18/ry/tv_va/checkpoint.pth.tar --benign --vanilla
