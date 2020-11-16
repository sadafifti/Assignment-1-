import cv2
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

#we will be using this function to display image later in notebook
def displayImage(img,t1):
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(t1)
    plt.show()
    
    
def channel(img,channel):
   # 0-b, 1-g, 2-r
    bgr_image = img[2].copy()
    bgr_image[:,:,channel] = 0 #empty blue channel
    return bgr_image



from skimage import io





def myConvolve2d(img, kernel):
   
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(img)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, img.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = img
    
    # Loop over every pixel of the image and implement convolution operation (element wise multiplication and summation). 
    # You can use two loops. The result is stored in the variable output.
    
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(img.shape[1]):
            # element-wise multiplication and summation 
            output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()
        
    
    return output