# CU-Net-Pytorch-
This is an unofficial code of Common and Unique information splitting network (CU-Net) with Pytorch. 


# Results
We train the CU-Net on the Cave dataset for the multispectral and RGB image fusion task (2x upscaling). The Cave contains 32 scenes and they are divide into three parts for training, testing and validation. In the training phase, we crop images into small patches, and there are 28512 pairs of samples in total.  CU-Net (number of blocks=4, kernel size=7) is trained by Adam over 200 epochs with a batch size of 32. The initial learning rate is 1e-4, and it is decayed by 0.9 every 50 epochs. It takes around 23 hours on a computer with 2080ti GPU.

Note: In their paper, the optimal kernel size is 8. However, in pytorch, an even kernel size cannot maintain the feature maps' height and width unchanged. Therefore, we set kernel size=7. 

# Train & Test
The train and test codes are available lines 7-107 and 113-148 of 'train_MMF.py'. 
