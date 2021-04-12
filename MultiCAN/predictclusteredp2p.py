from numpy import load
from numpy import zeros
from numpy import ones
from os import listdir
from numpy.random import randint
import numpy as np
import imageio
import cv2

import tensorflow as tf

import clusteredp2p

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

mpathtemplate = "./Models/models/gmodel_group{}_384001"
modelpaths = [mpathtemplate.format(0), mpathtemplate.format(1), \
    mpathtemplate.format(2), mpathtemplate.format(3), mpathtemplate.format(4)]

classifierpath = "./Models/ClusteredClassifier"

outputpath = "../data_imagenet/clustered_results/model5image{:0>4d}.png"

model = clusteredp2p.ClusteredP2PGan(5, classifierpath, (256,256,3), 1, *modelpaths)

data_ids = []
for filename in listdir("../data_imagenet/test/"):
    if filename.endswith('.npy'):
        data_ids.append(filename)

X1 = np.empty((len(data_ids), *(256,256,3)))
X2 = np.empty((len(data_ids), *(256,256,3)))

for i, ID in enumerate(data_ids):
    mergedImage = np.load('../data_imagenet/test/' + ID)
        
    X1[i,] = mergedImage[:, :256, :]
    X2[i,] = mergedImage[:, 256:, :]

bwimg = X1.copy()
X1 = (X1 - 127.5) / 127.5

num_batches = int(len(X1) / 250)

for i in range(num_batches):
    image_batch = X1[i*250:(i+1)*250]
    generatedImages = model.predict_multiple(image_batch)
    generatedImages = model.rescale_images(generatedImages)

    for k, image in enumerate(generatedImages):
        imageio.imwrite(outputpath.format(i*250+k), image)

image_batch = X1[num_batches*250:len(X1)]
generatedImages = model.predict_multiple(image_batch)
generatedImages = model.rescale_images(generatedImages)

for i, image in enumerate(generatedImages):
    imageio.imwrite(outputpath.format(num_batches*250+i), image)


