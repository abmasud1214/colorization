import p2pgan

import tensorflow as tf
from tensorflow import keras
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

import numpy as np
from os import listdir
import pickle

import time

import matplotlib.pyplot as plt

class ClusteredP2PGan():
    def __init__(self, num_categories, classifier_path, image_shape, mode=0, *model_paths):
        self.num_categories = num_categories
        self.image_shape = image_shape
        self.gan_models = []
        self.g_models = []
        self.d_models = []
        self.mode = mode
        self.file_names = []

        self.classifier = keras.models.load_model(classifier_path)

        if mode == 1:
            for path in model_paths:
                g_model = keras.models.load_model(path)
                self.g_models.append(g_model)
        elif mode == 0:
            for i in range(num_categories):
                d_model = p2pgan.define_discriminator(image_shape)
                g_model = p2pgan.define_generator(image_shape)
                gan_model = p2pgan.define_gan(g_model, d_model, image_shape)

                self.gan_models.append(gan_model)
                self.g_models.append(g_model)
                self.d_models.append(d_model)
    
    def rescale_images(self, images):
        images = (images + 1) * 255 / 2
        images = images.astype(np.uint8)
        return images
        

    def predict(self, image):
        classifier_predict = self.classifier.predict(image)
        image_category = np.argmax(classifier_predict)
        generatedImage = self.g_models[image_category].predict(image)
        return generatedImage
    
    def predict_multiple(self, images):
        # TODO: merge into predict
        classifier_predict = self.classifier.predict(images)
        image_category = np.argmax(classifier_predict, axis=1)
        generatedImages = None
        indices = []
        for i in range(len(self.g_models)):
            idx = np.where(image_category == i)[0]

            indices.extend(idx)
            model_images = images[idx]
            if len(model_images) == 0:
                continue
            if generatedImages is None:
                generatedImages = self.g_models[i].predict(model_images)
            else:
                im = self.g_models[i].predict(model_images)
                generatedImages = np.concatenate((generatedImages, im))

        sortedImages = np.full_like(generatedImages, 1)
        for i in range(len(generatedImages)):
            sortedImages[indices[i]] = generatedImages[i]
        return sortedImages
    
    def __load_file_names(self, path):
        self.file_names = []
        for filename in listdir(path + "/"):
            if filename.endswith('.npy'):
                self.file_names.append(filename)
    
    def __generate_real_samples(self, dataset, n_samples, patch_shape):
        # unpack dataset
        trainA, trainB = dataset
        # choose random instances
        ix = np.random.randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    def __load_sample_group(self, path, data_ids, dim, n_samples):
        ix = np.random.randint(0, len(data_ids), n_samples)
        X1 = np.empty((n_samples, *dim))
        X2 = np.empty((n_samples, *dim))
        
        for i, ID in enumerate(ix):
            mergedImage = np.load(path + '/' + data_ids[ID])
            
            X1[i,] = mergedImage[:, :256, :]
            X2[i,] = mergedImage[:, 256:, :]
        
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5

        return [X1, X2]    
    
    def __generate_fake_samples(self, g_model, samples, patch_shape):
        # generate fake instance
        X = g_model.predict(samples)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    def train(self, path, outputpath, n_epochs=5, n_batch=1, image_shape = (256, 256, 3), initial_step=0):
        if self.mode == 1:
            print("Cannot train with only generator models (Mode = 1)")
        
        n_patch = self.d_models[0].output_shape[1]

        self.__load_file_names(path)
        print("file names loaded")

        # bat_per_epo = int(len(self.file_names) / n_batch)
        # print(bat_per_epo)

        # n_steps = bat_per_epo * n_epochs
        
        samples = None
        samplegroupsize = 500

        d1loss = [[] for i in range(len(self.d_models))]
        d2loss = [[] for i in range(len(self.d_models))]
        gloss = [[] for i in range(len(self.g_models))]

        t1 = time.time()
        t = time.time()
        # cacheA = [None, None, None, None, None]
        # cacheB = [None, None, None, None, None]
        # cachey = [None, None, None, None, None]

        n_trains = [0, 0, 0, 0, 0]

        groups_per_epo = int(len(self.file_names) / samplegroupsize)
        print(groups_per_epo)

        n_steps = groups_per_epo * n_epochs

        for i in range(n_steps):
            samples = None
            samples = self.__load_sample_group(path, self.file_names, image_shape, samplegroupsize)
            classifier_predict = self.classifier.predict(samples[0])
            image_category = np.argmax(classifier_predict, axis=1)
            
            t2 = time.time()
            for m in range(self.num_categories):
                idx = np.where(image_category == m)[0]
                if len(idx) == 0:
                    continue
                mSamples = [samples[0][idx], samples[1][idx]]
                dl1, dl2, gl, t_s = self.__train_model(m, mSamples, n_batch, n_trains[m])
                d1loss[m].extend(dl1)
                d2loss[m].extend(dl2)
                gloss[m].extend(gl)
                n_trains[m] = t_s
                print("Trained model {}. {} batches. Time: {}".format(m, t_s, time.time()-t2))
                t2 = time.time()
            
            for s in range(len(d1loss)):
                if len(d1loss[s]) == 0:
                    print("{} not trained yet".format(s))
                    continue
                print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i*500, d1loss[s][-1], d2loss[s][-1], gloss[s][-1]))
            print("total time: {}, time group: {}".format(time.time() - t1, time.time() - t))
            t = time.time()
            print('n_trains:', n_trains)

            if (i+1)*samplegroupsize == 500 or (i+1)*samplegroupsize == 10000:
                self.__summarize_performance((i+1)*samplegroupsize, [d1loss, d2loss, gloss], samples, outputpath, 3)
            if (i+1) % groups_per_epo == 0:
                self.__summarize_performance((i+1)*samplegroupsize, [d1loss, d2loss, gloss], samples, outputpath, 3)

    def __train_model(self, model_num, samples, n_batch, prev_steps):
        g_model = self.g_models[model_num]
        d_model = self.d_models[model_num]
        gan_model = self.gan_models[model_num]

        batches = int(len(samples[0]) / n_batch)

        n_patch = d_model.output_shape[1]

        d1loss = []
        d2loss = []
        gloss = []

        for i in range(batches):
            [X_realA, X_realB], y_real = self.__generate_real_samples(samples, n_batch, n_patch)

            X_fakeB, y_fake = self.__generate_fake_samples(g_model, X_realA, n_patch)

            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            if (prev_steps + i + 1) % 10 == 0:
                d1loss.append(d_loss1)
                d2loss.append(d_loss2)
                gloss.append(g_loss)
        
        return (d1loss, d2loss, gloss, prev_steps + batches)

    def __summarize_performance(self, step, loss, dataset, destination, n_samples=3):
        # select a sample of input images
        [X_realA, X_realB], _ = self.__generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake sample
        X_fakeB = self.predict_multiple(X_realA)
        # scale all pixels from [-1,1] to [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0

        # plot real source images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(X_realA[i])

        # plot generated target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(X_fakeB[i])
        
        # plot real target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples*2 + i)
            plt.axis('off')
            plt.imshow(X_realB[i])

        filename1 = destination + '/plot_%06d.png' % (step+1)
        plt.savefig(filename1)
        plt.close()

        models_path = destination + '/models'

        # save generator models
        for i, model in enumerate(self.g_models):
            model.save(models_path + "/gmodel_group{}_{}".format(i, step+1))
        
        # save discriminator models
        for i, model in enumerate(self.d_models):
            model.save(models_path + "/dmodel_group{}_{}".format(i, step+1))
        
        for i in range(len(self.d_models)):
            with open(destination + "/data/d1error{}.data".format(i), 'wb') as d1file, \
                open(destination + '/data/d2error{}.data'.format(i), 'wb') as d2file, \
                open(destination + '/data/gerror{}.data'.format(i), 'wb') as gfile:
                pickle.dump(loss[0][i], d1file)
                pickle.dump(loss[1][i], d2file)
                pickle.dump(loss[2][i], gfile)
        
        print('Saved models and loss files at step {}.'.format(step+1))



        

  



        

