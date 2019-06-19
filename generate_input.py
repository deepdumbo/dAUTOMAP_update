import numpy as np
import cv2
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os
from PIL import Image


    
def load_images_from_folder(folder, n_cases,patch_size, mask_path, mask_type, mask_name,normalize=False, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    
    
    Modified by  Hanlu Yang, June/6/2019
    """

#    # Initialize the arrays:
#    if imrotate:  # number of images is 4 * n_im
#        bigy = np.empty((n_im * 4, 64, 64))
#        bigx = np.empty((n_im * 4, 64, 64, 2))
#    else:
#        bigy = np.empty((n_im, 64, 64))
#        bigx = np.empty((n_im, 64, 64, 2))

#    im = 0  # image counter
    bigy = []
    filenames = os.listdir(folder)

    for filename in filenames[n_cases[0]:n_cases[1]]:
        if not filename.startswith('.'):
            temp = loadmat(os.path.join(folder, filename))['res']
            print temp.shape
            # Clean the STONE sense recon data
            row, col = temp.shape
            temp = np.reshape(temp, (row, col, -1))
            #valid_mask = (np.abs(np.squeeze(temp[int(row/2), int(col/2), :])) != 0)
            #final_images = temp[:,:,valid_mask]
            final_images = temp
            
#            # Resize images
            #final_images = np.abs(final_images)
            final_images_resized = np.zeros((patch_size,patch_size,final_images.shape[2]))
            for i in range(final_images.shape[2]):
                final_images_resized[:,:,i] = cv2.resize(final_images[:,:,i], (patch_size,patch_size))
            
#            # Only take a small part of the data
#            final_images = final_images[140:180,140:180,:]
            
#            # Convert to abs values
#            final_images = np.abs(final_images)
#            
#            # Normalize based on single patient case
#            final_images = (final_images - np.mean(final_images)) / np.std(final_images)
            
#            bigy_temp = cv2.imread(os.path.join(folder, filename),
#                                   cv2.IMREAD_GRAYSCALE)
            
            
            bigy.append(final_images_resized)
    
    bigy = np.asarray(bigy)
    cases, row, col, imgs = bigy.shape
    bigy = np.transpose(np.reshape(np.transpose(bigy, (1,2,3,0)), (row, col, -1)), (2,0,1))
    
    # convert to k-space
    imgs, row, col = bigy.shape
    bigx = np.empty((imgs, row, col, 2))
    mask = read_mask(mask_path=mask_path,mask_type=mask_type,mask_name=mask_name,patch_size=patch_size,show_image=False)
    for i in range(imgs):
        bigx[i, :, :, :] = create_x(np.squeeze(bigy[i,:,:]),mask)
    
    # convert bigx from complex to abs values
    bigy = np.abs(bigy)
    
#            im += 1
#            if imrotate:
#                for angle in [90, 180, 270]:
#                    bigy_rot = im_rotate(bigy_temp, angle)
#                    bigx_rot = create_x(bigy_rot, normalize)
#                    bigy[im, :, :] = bigy_rot
#                    bigx[im, :, :, :] = bigx_rot
#                    im += 1

#        if imrotate:
#            if im > (n_im * 4 - 1):  # how many images to load
#                break
#        else:
#            if im > (n_im - 1):  # how many images to load
#                break

#    if normalize:
#        bigx = (bigx - np.amin(bigx)) / (np.amax(bigx) - np.amin(bigx))

    return bigx, bigy


def create_x_motion(y, normalize=False):
    """
    Prepares frequency data from image data: first image y is padded by 8
    pixels of value zero from each side (y_pad_loc1), then second image is
    created by moving the input image (64x64) 8 pixels down -> two same images
    at different locations are created; then both images are transformed to
    frequency space and their frequency space is combined as if the image
    moved half-way through the acquisition (upper part of freq space from one
    image and lower part of freq space from another image)
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: "Motion corrupted" frequency-space data of the input image,
    4D array of size (1, im_size1, im_size2, 2), third dimension (size: 2)
    contains real and imaginary part
    """

    # Pad y and move 8 pixels
    y_pad_loc1 = np.zeros((80, 80))
    y_pad_loc2 = np.zeros((80, 80))
    y_pad_loc1[8:72, 8:72] = y
    y_pad_loc2[0:64, 8:72] = y

    # FFT of both images
    img_f1 = np.fft.fft2(y_pad_loc1)  # FFT
    img_fshift1 = np.fft.fftshift(img_f1)  # FFT shift
    img_f2 = np.fft.fft2(y_pad_loc2)  # FFT
    img_fshift2 = np.fft.fftshift(img_f2)  # FFT shift

    # Combine halfs of both k-space - as if subject moved 8 pixels in the
    # middle of acquisition
    x_compl = np.zeros((80, 80), dtype=np.complex_)
    x_compl[0:41, :] = img_fshift1[0:41, :]
    x_compl[41:81, :] = img_fshift2[41:81, :]

    # Finally, separate into real and imaginary channels
    x_real = x_compl.real
    x_imag = x_compl.imag
    x = np.dstack((x_real, x_imag))

    x = np.expand_dims(x, axis=0)

    if normalize:
        x = x - np.mean(x)

    return x

def create_x(y, mask, motion=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    
    if motion:
        # Pad y and move 8 pixels
        y_pad_loc1 = np.zeros((80, 80))
        y_pad_loc2 = np.zeros((80, 80))
        y_pad_loc1[8:72, 8:72] = y
        y_pad_loc2[0:64, 8:72] = y
        
        # FFT of both images
        img_f1 = np.fft.fft2(y_pad_loc1)  # FFT
        img_fshift1 = np.fft.fftshift(img_f1)  # FFT shift
        img_f2 = np.fft.fft2(y_pad_loc2)  # FFT
        img_fshift2 = np.fft.fftshift(img_f2)  # FFT shift
        
        # Combine halfs of both k-space - as if subject moved 8 pixels in the
        # middle of acquisition
        x_compl = np.zeros((80, 80), dtype=np.complex_)
        x_compl[0:41, :] = img_fshift1[0:41, :]
        x_compl[41:81, :] = img_fshift2[41:81, :]
        
        # Finally, separate into real and imaginary channels
        x_real = x_compl.real
        x_imag = x_compl.imag
        x = np.dstack((x_real, x_imag))
        
        x = np.expand_dims(x, axis=0)
    else: 
        x = to_freq_space(y,mask)
        x = np.expand_dims(x, axis=0)

    return x

def read_mask(mask_path, mask_type, mask_name,patch_size,show_image=None):
    """
    read the mask, turn it to list whose value is [0,1]
    shape is [256 256]
    mask = 'cartes','gauss','radial','spiral'
    mask_name = 'mask_10/20/30/40/50/60/70/80/90'

    example; 
    #mask_list = read_mask(mask='cartes',mask_name='cartes_10',show_image=True)
    #print(mask_list)
    """
    path_test = mask_path

    mask= Image.open(path_test+"/"+"{}".format(mask_type)+
                "/"+"{}.tif".format(mask_name))
    mask_list = np.asarray(list (mask.getdata() ))

    mask_list = mask_list / np.amax(mask_list)
    #either use from future or use // to get float result
    mask_list = np.reshape(mask_list,(patch_size,patch_size))
    if (show_image == True):

        print(mask_list.shape)
        plt.figure()
        plt.imshow(mask_list,cmap='gray')
        plt.show()
        print(mask_list)
    return mask_list




def to_freq_space(img,mask):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """
    img_f = np.fft.fft2(img)  # FFT
    #print('img_f = ', img_f)
    img_undersample = img_f * ( mask)
    #print('img_under = ', img_undersample)
    #plt.figure()
    #plt.imshow(np.abs(img_undersample),cmap='gray')
    #plt.show()
    img_fshift = np.fft.fftshift(img_undersample)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """
    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated


'''
# For debugging: show the images and their frequency space

dir_temp = 'path to folder with images'
X, Y = load_images_from_folder(dir_temp, 5, normalize=False, imrotate=True)

print(Y.shape)
print(X.shape)


plt.subplot(221), plt.imshow(Y[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(Y[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Y[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(Y[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

X_m = 20*np.log(np.sqrt(np.power(X[:, :, :, 0], 2) +
                        np.power(X[:, :, :, 1], 2)))  # Magnitude
plt.subplot(221), plt.imshow(X_m[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(X_m[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(X_m[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(X_m[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
'''
