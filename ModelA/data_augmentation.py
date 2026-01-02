from skimage.util import random_noise
import numpy as np


def augment_img(img):
    # Implement data augmentation logic here


    augmented = []

    # Introduce table to compare augmentations (is this useful?)
    # Add original image (to augment data)
    augmented.append(img)
  

    # Flip image
    augmented.append(np.fliplr(img))

    # Add Gaussian noise
    augmented.append(random_noise(img, mode='gaussian', var=0.01))#gaussian noise

   # augmented.append(rotate(img, 15, mode='edge'))
   # augmented.append(gaussian(img, sigma=1.0)) add blur
   # augmented.append(np.clip(img * 1.2, 0, 1)) add brightness
   # augmented.append(np.clip((img-np.mean(img)) * 1.5 + np.mean(img), 0, 1)) adjust contrast

# BEST ACCURACY OBTAINED WITH FLIP + contrast/blur/gaussian noise --> find explanation and include in report (only two elements seem to be enough)
    return augmented
    

    