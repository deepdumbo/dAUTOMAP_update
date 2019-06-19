import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from generate_input import load_images_from_folder
import models
import os; 

""" Loads n_im images from the folder, and choose 4 of them as 
	present example 
    Modified by Hanlu Yang, June/6/2019
    """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


## Load development/test data:
dir_dev = "/data/Catherine/Deep_MRI_Unet/AUTOMAP/test_AUTOMAP/test_data"

log_number = 1950 # choose the model number you want to reload 

#training parameters 
patch_size = 256
mask_path = os.path.join(os.path.expanduser('/'),'home','tuh21221','Documents','PythonFile','mask')
mask_type='spiral'
mask_name='spiral_10'

##n_im_dev = 60  # How many images to load
## Load images and create motion-corrupted frequency space
## No normalization or rotations:
n_cases = (0,20)  # load image data from 0 to 2 
X_dev, Y_dev = load_images_from_folder(  # Load images for training
    dir_dev,
    n_cases,
    patch_size=patch_size,
    mask_path=mask_path, 
    mask_type=mask_type, 
    mask_name=mask_name,
    normalize=False,
    imrotate=False)


X_dev_R = X_dev[:, :, :, 0]
X_dev_R = torch.from_numpy(np.array(X_dev_R)).float()
X_dev_R = torch.unsqueeze(X_dev_R, 1)

X_dev_I = X_dev[:, :, :, 1]
X_dev_I = torch.from_numpy(np.array(X_dev_I)).float()
X_dev_I = torch.unsqueeze(X_dev_I, 1)

Y_dev = torch.from_numpy(Y_dev).float()
Y_dev = torch.unsqueeze(Y_dev, 1)
#Y_train = torch.cat((Y_train,Y_train),dim=1)

X_dev = torch.cat((X_dev_R,X_dev_I),dim=1)
X_dev,Y_dev = Variable(X_dev, requires_grad=False),Variable(Y_dev, requires_grad=False)
X_dev,Y_dev = X_dev.to(device),Y_dev.to(device)

print('X_dev.shape at input = ', X_dev.shape)
print('Y_dev.shape at input = ', Y_dev.shape)

recon_test = torch.load('./checkpoint/dAUTOMAP_{}.pkl'.format(log_number))
recon_test.to(device)


#print('Y_recon.shape = ', Y_recon.shape)


# Visualize the images, their reconstruction using iFFT and using trained model
# 4 images to visualize:
im1 = 15
im2 = 16
im3 = 17
im4 = 18

Y_recon = recon_test(X_dev)
# Y_recon2 = recon_test(X_dev[im2,:,:, :])
# Y_recon3 = recon_test(X_dev[im3,:,:, :])
# Y_recon4 = recon_test(X_dev[im4,:,:, :])
# iFFT back to image from corrupted frequency space
# Complex image from real and imaginary part
X_dev = X_dev.cpu()
X_dev_compl =X_dev[:, 0, :, :].numpy()+ X_dev[:, 1, :, :].numpy() * 1j

#iFFT
X_iFFT0 = np.fft.ifft2(X_dev_compl[im1, :, :])
X_iFFT1 = np.fft.ifft2(X_dev_compl[im2, :, :])
X_iFFT2 = np.fft.ifft2(X_dev_compl[im3, :, :])
X_iFFT3 = np.fft.ifft2(X_dev_compl[im4, :, :])

# Magnitude of complex image
X_iFFT_M1 = np.sqrt(np.power(X_iFFT0.real, 2)
                    + np.power(X_iFFT0.imag, 2))
X_iFFT_M2 = np.sqrt(np.power(X_iFFT1.real, 2)
                    + np.power(X_iFFT1.imag, 2))
X_iFFT_M3 = np.sqrt(np.power(X_iFFT2.real, 2)
                    + np.power(X_iFFT2.imag, 2))
X_iFFT_M4 = np.sqrt(np.power(X_iFFT3.real, 2)
                    + np.power(X_iFFT3.imag, 2))

# SHOW
# Show Y - input images
Y_dev = Y_dev[:,0, :, :]
Y_recon = Y_recon.cpu()
Y_recon = Y_recon[:,0, :, :]
Y_recon = Y_recon.detach().numpy()


# Y_recon1 = Y_recon1.cpu()
# Y_recon1 = Y_recon1[0, :, :]
# Y_recon1 = Y_recon1.detach().numpy()

# Y_recon2 = Y_recon2.cpu()
# Y_recon2 = Y_recon2[0, :, :]
# Y_recon2 = Y_recon2.detach().numpy()

# Y_recon3 = Y_recon3.cpu()
# Y_recon3 = Y_recon3[0, :, :]
# Y_recon3 = Y_recon3.detach().numpy()

# Y_recon4 = Y_recon4.cpu()
# Y_recon4 = Y_recon4[0, :, :]
# Y_recon4 = Y_recon4.detach().numpy()


plt.subplot(341), plt.imshow(Y_dev[im1, :, :], cmap='gray')
plt.title('Y_dev1'), plt.xticks([]), plt.yticks([])
plt.subplot(342), plt.imshow(Y_dev[im2, :, :], cmap='gray')
plt.title('Y_dev2'), plt.xticks([]), plt.yticks([])
plt.subplot(343), plt.imshow(Y_dev[im3, :, :], cmap='gray')
plt.title('Y_dev3'), plt.xticks([]), plt.yticks([])
plt.subplot(344), plt.imshow(Y_dev[im4, :, :], cmap='gray')
plt.title('Y_dev4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using iFFT
plt.subplot(345), plt.imshow(X_iFFT_M1, cmap='gray')
plt.title('X_iFFT1'), plt.xticks([]), plt.yticks([])
plt.subplot(346), plt.imshow(X_iFFT_M2, cmap='gray')
plt.title('X_iFFT2'), plt.xticks([]), plt.yticks([])
plt.subplot(347), plt.imshow(X_iFFT_M3, cmap='gray')
plt.title('X_iFFT3'), plt.xticks([]), plt.yticks([])
plt.subplot(348), plt.imshow(X_iFFT_M4, cmap='gray')
plt.title('X_iFFT4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using model
# plt.subplot(349), plt.imshow(Y_recon, cmap='gray')
# plt.title('Y_recon1'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 4, 10), plt.imshow(Y_recon2, cmap='gray')
# plt.title('Y_recon2'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 4, 11), plt.imshow(Y_recon3, cmap='gray')
# plt.title('Y_recon3'), plt.xticks([]), plt.yticks([])
# plt.subplot(3, 4, 12), plt.imshow(Y_recon4, cmap='gray')
# plt.title('Y_recon4'), plt.xticks([]), plt.yticks([])
# plt.subplots_adjust(hspace=0.3)
# plt.show()
plt.subplot(349), plt.imshow(Y_recon[im1, :, :], cmap='gray')
plt.title('Output-im1'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 10), plt.imshow(Y_recon[im2, :, :], cmap='gray')
plt.title('Output-im2'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 11), plt.imshow(Y_recon[im3, :, :], cmap='gray')
plt.title('Output-im3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 12), plt.imshow(Y_recon[im4, :, :], cmap='gray')
plt.title('Output-im4'), plt.xticks([]), plt.yticks([])
plt.subplots_adjust(hspace=0.3)
plt.show()

# Chong Duan - Display resutls
# Show X - input k-space
plt.subplot(341), plt.imshow(np.abs(X_dev_compl[im1, :, :]), cmap='gray')
plt.title('Input-im1'), plt.xticks([]), plt.yticks([])
plt.subplot(342), plt.imshow(np.abs(X_dev_compl[im2, :, :]), cmap='gray')
plt.title('Input-im2'), plt.xticks([]), plt.yticks([])
plt.subplot(343), plt.imshow(np.abs(X_dev_compl[im3, :, :]), cmap='gray')
plt.title('Input-im3'), plt.xticks([]), plt.yticks([])
plt.subplot(344), plt.imshow(np.abs(X_dev_compl[im4, :, :]), cmap='gray')
plt.title('Input-im4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using iFFT
plt.subplot(345), plt.imshow(X_iFFT_M1, cmap='gray')
plt.title('iFFT_im1'), plt.xticks([]), plt.yticks([])
plt.subplot(346), plt.imshow(X_iFFT_M2, cmap='gray')
plt.title('iFFT_im2'), plt.xticks([]), plt.yticks([])
plt.subplot(347), plt.imshow(X_iFFT_M3, cmap='gray')
plt.title('iFFT_im3'), plt.xticks([]), plt.yticks([])
plt.subplot(348), plt.imshow(X_iFFT_M4, cmap='gray')
plt.title('iFFT_im4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using model
plt.subplot(349), plt.imshow(Y_recon[im1, :, :], cmap='gray')
plt.title('Output-im1'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 10), plt.imshow(Y_recon[im2, :, :], cmap='gray')
plt.title('Output-im2'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 11), plt.imshow(Y_recon[im3, :, :], cmap='gray')
plt.title('Output-im3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 12), plt.imshow(Y_recon[im4, :, :], cmap='gray')
plt.title('Output-im4'), plt.xticks([]), plt.yticks([])
plt.subplots_adjust(hspace=0.3)
plt.show()