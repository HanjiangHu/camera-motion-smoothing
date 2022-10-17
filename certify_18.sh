
python certify.py metaroom models/metaroom/resnet18/tz/tv_noise_0.1/checkpoint.pth.tar 0.1 resolvable_tz data/predict/metaroom/resnet18/tz/tv_noise_0.1 --batch 200
python certify.py metaroom models/metaroom/resnet18/ty/tv_noise_0.05/checkpoint.pth.tar 0.05 resolvable_ty data/predict/metaroom/resnet18/ty/tv_noise_0.05 --batch 200
python certify.py metaroom models/metaroom/resnet18/tx/tv_noise_0.05/checkpoint.pth.tar 0.05 resolvable_tx data/predict/metaroom/resnet18/tx/tv_noise_0.05 --batch 200
python certify.py metaroom models/metaroom/resnet18/rx/tv_noise_2.5/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/metaroom/resnet18/rx/tv_noise_2.5 --batch 200
python certify.py metaroom models/metaroom/resnet18/ry/tv_noise_2.5/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/metaroom/resnet18/ry/tv_noise_2.5 --batch 200
python certify.py metaroom models/metaroom/resnet18/rz/tv_noise_7/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/metaroom/resnet18/rz/tv_noise_7 --batch 200

python certify.py metaroom models/metaroom/resnet18/tx/tv_va/checkpoint.pth.tar 0.05 resolvable_tx data/predict/metaroom/resnet18/tx/tv_va --batch 200 --vanilla
python certify.py metaroom models/metaroom/resnet18/ty/tv_va/checkpoint.pth.tar 0.05 resolvable_ty data/predict/metaroom/resnet18/ty/tv_va --batch 200 --vanilla
python certify.py metaroom models/metaroom/resnet18/tz/tv_va/checkpoint.pth.tar 0.1 resolvable_tz data/predict/metaroom/resnet18/tz/tv_va --batch 200 --vanilla
python certify.py metaroom models/metaroom/resnet18/rx/tv_va/checkpoint.pth.tar 0.04363323 resolvable_rx data/predict/metaroom/resnet18/rx/tv_va --batch 200 --vanilla
python certify.py metaroom models/metaroom/resnet18/ry/tv_va/checkpoint.pth.tar 0.04363323 resolvable_ry data/predict/metaroom/resnet18/ry/tv_va --batch 200 --vanilla
python certify.py metaroom models/metaroom/resnet18/rz/tv_va/checkpoint.pth.tar 0.122173 resolvable_rz data/predict/metaroom/resnet18/rz/tv_va --batch 200 --vanilla



