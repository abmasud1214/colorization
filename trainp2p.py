from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import numpy as np

from os import listdir
import os

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
import matplotlib.pyplot as pyplot

import time

import pickle

import p2pgan

def load_file_names(path):
    data_ids = []
    for filename in listdir(path + "/"):
        if filename.endswith('.npy'):
            data_ids.append(filename)
    return data_ids

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def load_sample_group(path, data_ids, dim, n_samples):
    ix = randint(0, len(data_ids), n_samples)
    X1 = np.empty((n_samples, *dim))
    X2 = np.empty((n_samples, *dim))
    
    for i, ID in enumerate(ix):
        mergedImage = np.load(path + '/' + data_ids[ID])
        
        X1[i,] = mergedImage[:, :256, :]
        X2[i,] = mergedImage[:, 256:, :]
    
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]        

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, loss, dataset, destination, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = destination + '/plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = destination + '/model_%06d' % (step+1)
    filenamed = destination + '/modeld_%06d' % (step+1)
    
    g_model.save(filename2)
    d_model.save(filenamed)

    with open(destination + '/d1error{}.data'.format(step+1), 'wb') as d1file, open(destination + '/d2error{}.data'.format(step+1), 'wb') as d2file, \
        open(destination + '/gerror{}.data'.format(step+1), 'wb') as gfile:
        pickle.dump(loss[0], d1file)
        pickle.dump(loss[1], d2file)
        pickle.dump(loss[2], gfile)

    print('>Saved: %s and %s' % (filename1, filename2))    

# train pix2pix models
def train(d_model, g_model, gan_model, path, outputpath, n_epochs=100, n_batch=1, image_shape = (256, 256, 3), initial_step = 0):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    data_ids = load_file_names(path)
    print("file names loaded")
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(data_ids) / n_batch)
    print(bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    
    samples = None
    samplegroupsize = 1500

    d1loss = []
    d2loss = []
    gloss = []

    t1 = time.time()
    t = time.time()
    for i in range(n_steps):
        # Load samplegroupsize images into RAM after previous group has been used.
        if (i*n_batch) % samplegroupsize == 0:
            samples = None
            samples = load_sample_group(path, data_ids, image_shape, samplegroupsize)

        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(samples, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # Save loss every 10 iterations for loss graph.
        if i % 10 == 0:
            d1loss.append(d_loss1)
            d2loss.append(d_loss2)
            gloss.append(g_loss)
        total_steps = (i + initial_step)
        # summarize performance every 100 iterations.
        if (i+1) % 100 == 0:
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (total_steps+1, d_loss1, d_loss2, g_loss))
            print("total time: {}, time 100: {}".format(time.time() - t1, time.time() - t))
            t = time.time()
        # summarize model performance
        if (i+1) == 100 or (i+1) == 500 or (i+1) == 1000:
            summarize_performance(total_steps, g_model, d_model, [d1loss, d2loss, gloss], samples, outputpath, 3)
        if (i+1) % (bat_per_epo) == 0:
            summarize_performance(total_steps, g_model, d_model, [d1loss, d2loss, gloss], samples, outputpath, 3)
    
if __name__ == "__main__":
    #enable gpu
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

    image_shape = (256, 256, 3)
    path = "./data_imagenet/train"

    # # define the models
    d_model = p2pgan.define_discriminator(image_shape)
    g_model = p2pgan.define_generator(image_shape)
    gan_model = p2pgan.define_gan(g_model, d_model, image_shape)

    # define the models (load models)
    # d_model = tf.keras.models.load_model('./256_gpumodel4/modeld_005525')
    # g_model = tf.keras.models.load_model('./256_gpumodel4/model_005525')
    # gan_model = p2pgan.define_gan(g_model, d_model, image_shape)

    train(d_model, g_model, gan_model, path, "./in_gpumodel2", n_epochs = 5, n_batch = 25, initial_step=0)

    