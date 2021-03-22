from os import listdir
from numpy import asarray
from numpy import vstack
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cv2
from numpy import savez_compressed
import random
# import imageio

# load all images in a directory into memory
def load_images(path, size=(256,256)):
    # enumerate filenames in directory, assume all are images
    trainnum = 0
    testnum = 0
    for foldername in listdir(path):
        for filename in listdir(path + "/" + foldername):
            # load and resize the image
            pixels = cv2.imread(path + "/" + foldername + "/" + filename)
            if pixels is None:
                continue
            if size[0] - pixels.shape[0] < 0 or size[1] - pixels.shape[1] < 0:
                if pixels.shape[0] >= pixels.shape[1]:
                    pixels = cv2.resize(pixels, dsize=(int((pixels.shape[1] / pixels.shape[0]) * size[1]), size[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    pixels = cv2.resize(pixels, dsize=(size[1], int((pixels.shape[0] / pixels.shape[1])*size[0])), interpolation=cv2.INTER_NEAREST)
            pixelsResized = cv2.copyMakeBorder(pixels, 0, size[0] - pixels.shape[0], 0, size[1] - pixels.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])

            color_img = pixelsResized
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            
            mergedImage = np.zeros([size[1], size[0]*2, 3], dtype=np.uint8)
            mergedImage[:,256:,:] = color_img
            mergedImage[:, :256,:] = gray_img_3ch
            
            if random.random() < 0.99:
#                 imageio.imwrite("./data/train/trainimage{:0>4d}.png".format(trainnum), mergedImage)
                np.save('./data_imagenet/train/trainimage{:0>6d}.npy'.format(trainnum), mergedImage)
                trainnum += 1
            else:
#                 imageio.imwrite("./data/test/testimage{:0>4d}.png".format(testnum), mergedImage)
                np.save('./data_imagenet/test/testimage{:0>6d}.npy'.format(testnum), mergedImage)
                testnum += 1
            
        print("finished folder " + foldername)
#     return [asarray(src_list), asarray(tar_list)]

# dataset path
path = "../ImageNet-Datasets-Downloader/250ImageNet"
# load dataset
load_images(path)