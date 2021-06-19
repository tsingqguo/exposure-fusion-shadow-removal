import os

import cv2
import matplotlib.pyplot as plt

import PIL.Image as Image
from skimage.filters import threshold_otsu
from skimage import filters
import numpy as np
from skimage.morphology import disk

# get mask from shadow image and shadow free image

train_A_path = '../data/ISTD+/test_A'
train_C_path = '../data/ISTD+/test_C'
root_path = os.listdir(train_A_path)

for file in root_path:
    s_name = os.path.join(train_A_path, file)
    sf_name = os.path.join(train_C_path, file)

    s_img = Image.open(s_name).convert("L")
    s_img = np.array(s_img).astype(np.float32)
    sf_img = Image.open(sf_name).convert("L")
    sf_img = np.array(sf_img).astype(np.float32)
    diff = (np.asarray(sf_img, dtype='float32') - np.asarray(s_img, dtype='float32'))
    # diff[diff < 0] = 0
    L = threshold_otsu(diff)
    mask = np.float32(diff > L) * 255
    mask = filters.median(mask, disk(5))
    mask = Image.fromarray(mask).convert("L")
    # plt.imshow(mask)
    # plt.show()
    mask.save('../data/ISTD+/test_mask/' + file)
