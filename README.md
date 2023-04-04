# Robustness Certification via Camera Motion Smoothing

This the official code,dataset and models for CoRL 2022 "Robustness Certification of Visual Perception Models via Camera Motion Smoothing". Check out [arxiv PDF](https://arxiv.org/abs/2210.04625) for more details.

[![Robustness Certification of Visual Perception Models via Camera Motion Smoothing](https://res.cloudinary.com/marcomontalbano/image/upload/v1665985036/video_to_markdown/images/youtube--iCfRBk3O3CA-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=iCfRBk3O3CA "Robustness Certification of Visual Perception Models via Camera Motion Smoothing")

## Preparation
Run the following command to install all packages.

``pip install torchvision seaborn numpy scipy setproctitle matplotlib pandas statsmodels opencv_python torch Pillow python_dateutil setGPU numba open3d cupy-cuda116 tqdm``


## MetaRoom dataset
Download the dataset from [here](https://drive.google.com/file/d/1rX-21GtWRxpJsnjb9D2wCwtdwqnXEM8a/view?usp=sharing) which includes camera motion augmented and vanilla dataset with 6 camera motions. Unzip the dataset  in the root path for the `./metaroom/metaroom_XXX` folder structure. Under each dataset, there are training, validation set of images and certification set of point clouds, camera poses and intrinsics as pickle files for test. Note that training and validation set are captured directly from Webots while certification set are projected from point clouds to make the comparison between empirical robustness and certifiable robustness fair. You can also download the demo point cloud of the entire room from [here](https://drive.google.com/file/d/1X5y1vIrDTRUFkbUEkGaxDTLMNSHSNyCw/view?usp=sharing) for visualization. 




## Model training of motion augmented and vanilla models
For the model training, run the `bash ./scripts/train_18.sh` and `bash ./scripts/train_50.sh` for ResNet-18 and ResNet-50 architectures for both vanilla and motion augmented model training. 


## Benign and empirical robust accuracy
For benign and empirical robust accuracy of  both base vanilla and base motion augmented models, run `bash ./scripts/empirical_test_18_base.sh` and `bash ./scripts/empirical_test_50_base.sh`. 
To implement motion smoothing for both vanilla and motion augmented models to get benign and empirical robust accuracy, run `bash ./scripts/empirical_test_18.sh` and `bash ./scripts/empirical_test_50.sh`. Note that `--uniform` indicates the empirical robust accuracy while the default is benign accuracy. Change `--pretrain` for correct pretrained models if necessary. Change `--sample_num` (default 100) to adjust grid search based worst case robustness if necessary.

## Certified accuracy for smoothed models
For certification of smoothed vanilla and motion augmented models, run the `bash ./scripts/certify_18.sh` and `bash ./scripts/certify_18.sh`  and output logs are located in `./data`.   Change `--pretrain` for correct pretrained models if necessary.
To get the certified accuracy, run the `bash ./scripts/analyze_18.sh` and `bash ./scripts/analyze_18.sh` and output logs are located in `./data/results`. The figures of certified accuracy of radius can be plotted by running `python plot.py`.


## Citation
If you use this code in your own work, please cite this paper:

H. Hu, Z. Liu, L. Li, J. Zhu and D. Zhao
"[Robustness Certification of Visual Perception Models via Camera Motion Smoothing](https://arxiv.org/abs/2210.04625)", CoRL 2022

```
@InProceedings{pmlr-v205-hu23b,
  title = 	 {Robustness Certification of Visual Perception Models via Camera Motion Smoothing},
  author =       {Hu, Hanjiang and Liu, Zuxin and Li, Linyi and Zhu, Jiacheng and Zhao, Ding},
  booktitle = 	 {Proceedings of The 6th Conference on Robot Learning},
  pages = 	 {1309--1320},
  year = 	 {2023},
  volume = 	 {205},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {14--18 Dec},
  publisher =    {PMLR}
}
```

## Reference
> - [TSS](https://github.com/AI-secure/semantic-randomized-smoothing)
> - [Randomized Smoothing](https://github.com/locuslab/smoothing)
