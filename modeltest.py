from numpy import load
from numpy import zeros
from numpy import ones
from os import listdir
from numpy.random import randint
import numpy as np
import imageio
import cv2

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot

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

g_model = tf.keras.models.load_model('./in_gpumodel2/model_051220')
g_model.summary()

data_ids = []
for filename in listdir("./data_imagenet/test/"):
    if filename.endswith('.npy'):
        data_ids.append(filename)

X1 = np.empty((len(data_ids), *(256,256,3)))
X2 = np.empty((len(data_ids), *(256,256,3)))

for i, ID in enumerate(data_ids):
    mergedImage = np.load('data_imagenet/test/' + ID)
        
    X1[i,] = mergedImage[:, :256, :]
    X2[i,] = mergedImage[:, 256:, :]

bwimg = X1.copy()
X1 = (X1 - 127.5) / 127.5
# X2 = (X2 - 127.5) / 127.5    

X1_3 = X1[:3]
    
generatedImages = g_model.predict(X1[:1000])
generatedImages = (generatedImages + 1) * 255 / 2
generatedImages = generatedImages.astype(np.uint8)

for i, image in enumerate(generatedImages):
    mergedImage = np.zeros([256, 256*3, 3], dtype=np.uint8)
    mergedImage[:,:256,:] = bwimg[i]
    mergedImage[:,256:256*2,:] = image
    mergedImage[:, 256*2:,:] = X2[i]
    imageio.imwrite("./data_imagenet/merged_test2_in5/mergedimage{:0>6d}.png".format(i), mergedImage)

generatedImages = g_model.predict(X1[1000:2000])
generatedImages = (generatedImages + 1) * 255 / 2
generatedImages = generatedImages.astype(np.uint8)

for i, image in enumerate(generatedImages):
    ix = i + 1000
    mergedImage = np.zeros([256, 256*3, 3], dtype=np.uint8)
    mergedImage[:,:256,:] = bwimg[ix]
    mergedImage[:,256:256*2,:] = image
    mergedImage[:, 256*2:,:] = X2[ix]
    imageio.imwrite("./data_imagenet/merged_test2_in5/mergedimage{:0>6d}.png".format(ix), mergedImage)

generatedImages = g_model.predict(X1[2000:])
generatedImages = (generatedImages + 1) * 255 / 2
generatedImages = generatedImages.astype(np.uint8)

for i, image in enumerate(generatedImages):
    ix = i + 2000
    mergedImage = np.zeros([256, 256*3, 3], dtype=np.uint8)
    mergedImage[:,:256,:] = bwimg[ix]
    mergedImage[:,256:256*2,:] = image
    mergedImage[:, 256*2:,:] = X2[ix]
    imageio.imwrite("./data_imagenet/merged_test2_in5/mergedimage{:0>6d}.png".format(ix), mergedImage)

# generatedImages = g_model.predict(X1[:5])

# generatedImages = g_model.predict(X1)
# generatedImages = (generatedImages + 1) * 255 / 2
# generatedImages = generatedImages.astype(np.uint8)

# for i, image in enumerate(generatedImages):
#     mergedImage = np.zeros([256, 256*3, 3], dtype=np.uint8)
#     mergedImage[:,:256,:] = bwimg[i]
#     mergedImage[:,256:256*2,:] = image
#     mergedImage[:, 256*2:,:] = X2[i]
#     imageio.imwrite("./data_256/merged_test_256/mergedimage{:0>4d}.png".format(i), mergedImage)
