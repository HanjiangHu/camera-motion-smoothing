python train.py metaroom resnet50 translation_z models/metaroom/resnet50/tz/tv_noise_0.1 --batch 32   --lr 0.001 --gpu 0
 python train.py metaroom resnet50 translation_x models/metaroom/resnet50/tx/tv_noise_0.05 --batch 32   --lr 0.001 --gpu 0
 python train.py metaroom resnet50 translation_y models/metaroom/resnet50/ty/tv_noise_0.05 --batch 32   --lr 0.001 --gpu 0
 python train.py metaroom resnet50 rotation_z models/metaroom/resnet50/rz/tv_noise_7 --batch 32   --lr 0.001 --gpu 0
 python train.py metaroom resnet50 rotation_x models/metaroom/resnet50/rx/tv_noise_2.5 --batch 32   --lr 0.001 --gpu 0
 python train.py metaroom resnet50 rotation_y models/metaroom/resnet50/ry/tv_noise_2.5 --batch 32   --lr 0.001 --gpu 0

 python train.py metaroom resnet50 translation_z models/metaroom/resnet50/tz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100
 python train.py metaroom resnet50 translation_x models/metaroom/resnet50/tx/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100
 python train.py metaroom resnet50 translation_y models/metaroom/resnet50/ty/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100
 python train.py metaroom resnet50 rotation_x models/metaroom/resnet50/rx/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100
 python train.py metaroom resnet50 rotation_y models/metaroom/resnet50/ry/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100
 python train.py metaroom resnet50 rotation_z models/metaroom/resnet50/rz/tv_va --batch 32 --noise_sd 0.0  --lr 0.001 --gpu 0 --vanilla --epochs 100




