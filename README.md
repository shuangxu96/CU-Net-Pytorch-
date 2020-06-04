# CU-Net-Pytorch-
This is an unofficial code of Common and Unique information splitting network (CU-Net) with Pytorch. The offcial code (Tensorflow version) is [here](https://github.com/cindydeng1991/TPAMI-CU-Net).

## Requirements
- pytorch (my version is 1.3.0)
- kornia (used for computing PSNR and SSIM)
- tensorboardX
- h5py

## Results
We train the CU-Net on the Cave dataset for the multispectral and RGB image fusion task (2x upscaling). The Cave contains 32 scenes and they are divide into three parts for training, testing and validation. In the training phase, we crop images into small patches, and there are 28512 pairs of samples in total.  CU-Net (number of blocks=4, kernel size=7) is trained by Adam over 200 epochs with a batch size of 32. The initial learning rate is 1e-4, and it is decayed by 0.9 every 50 epochs. It takes around 23 hours on a computer with 2080ti GPU.

|Metrics|R&F apples|R&F peppers|sponges|stuffed_toys|superballs|thread_spools|Mean|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|PSNR|48.66395569|46.91352844|42.19599152|42.60326385|45.60707855|43.44329834|45.09509277|
|SSIM|0.98826766|0.990578771|0.988602698|0.983907402|0.982619643|0.981697559|0.971616209|

Note: In their paper, the optimal kernel size is 8. However, in pytorch, an even kernel size cannot maintain the feature maps' height and width unchanged. Therefore, we set kernel size=7. 


## Train & Test
### Retrain and Test CU-Net
The train and test codes are available lines 7-107 and 113-148 of `train_MMF.py`. If you want to retrain this network, you should:
- Please download and unzip the [dataset](https://mega.nz/folder/LQwVhZ4J#PNGzSnjkrqjPD4M7Td2jMA). My folder is organized as follows:
```
    mypath
    ├── train
    │   ├── balloons.mat 
    │   ├── beads.mat
    │   └── ...
    ├── test
    │   ├── real_and_fake_apples.mat
    │   ├── real_and_fake_peppers.mat
    │   └── ...
    ├── validation
    │   ├── paints.mat
    │   ├── photo_and_face.mat
    │   └── ...
    └── ...
```

- In line 39, set `prepare_data_flag` to `True`. This variable controls whether prepare the training set as an H5 file. Once you have prepared the H5 file before, please set it to `False`, otherwise it will waste a long time to recreate the H5 file.
- In lines 41, 47, and 115, input the paths of the training, validation and testing datasets.
- Run lines 7-107 for training.
- Run lines 113-148 for testing.

### Test CU-Net with Pretrained Weights
A pretrained weight file is provided. If you do not want to retrain this model, please run `test_MMF.py`, in line 39 of which you should input the path of the testing dataset. 

## Reference
```
@inproceedings{Deng2019deep,
author = {Deng, Xin and Dragotti, Pier Luigi},
title = {Deep Convolutional Neural Network for Multi-modal Image Restoration and Fusion},
booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
year= {2020}
}
```
