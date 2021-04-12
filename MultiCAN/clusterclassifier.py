import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
from os import listdir

import matplotlib.pyplot as plt

def create_model(image_shape, num_categories):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1), \
        activation='relu',input_shape=(image_shape)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1), \
        activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(1,1), \
        activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(.5))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(.5))

    model.add(Dense(num_categories, activation='softmax'))

    model.compile(loss='categorical_crossentropy', \
        optimizer=Adam(), \
        metrics=['accuracy'])
    
    model.summary()
    return model

class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, keys, image_filenames, labels, path, batch_size=16, \
        dim=(256,256), n_channels=3, n_classes=5, shuffle=True):
        self.keys = keys
        self.image_filenames = image_filenames
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.path = path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return (np.ceil(len(self.keys) / float(self.batch_size))).astype(np.int)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.keys))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index) :
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_ID_temp = [self.keys[k] for k in indexes]

        X, y = self.__data_generation(list_ID_temp)

        return X, y

    def __data_generation(self, list_ID_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_ID_temp):
            filename = self.image_filenames[ID]
            label = self.labels[ID]
            l = int(label[6])
            X[i,] = np.load('{}/{}/{}'.format(self.path, label, filename))
            y[i] = l

        X = (X - 127.5) / 127.5
                
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

if __name__ == "__main__":
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
    

    filenames = dict()
    labels = dict()
    train_keys = []
    test_keys = []

    train_path = '../clustereddata/train'
    test_path = '../clustereddata/val'

    save_path = './Models'

    key = 0
    for group_folder in listdir(train_path + "/"):
        for filename in listdir(train_path + "/" + group_folder):
            if filename.endswith('.npy'):
                filenames[key] = filename
                labels[key] = group_folder
                train_keys.append(key)
                key += 1
    
    for group_folder in listdir(test_path + "/"):
        for filename in listdir(test_path + "/" + group_folder):
            if filename.endswith('.npy'):
                filenames[key] = filename
                labels[key] = group_folder
                test_keys.append(key)
                key += 1

    training_generator = ImageGenerator(train_keys, filenames, labels, train_path, \
        batch_size=16, dim=(256,256), n_channels=3, n_classes=5, shuffle=True)

    validation_generator = ImageGenerator(test_keys, filenames, labels, test_path, \
        batch_size=16, dim=(256,256), n_channels=3, n_classes=5, shuffle=True)

    model = create_model((256,256,3), 5)

    history = model.fit(x=training_generator, 
        validation_data=validation_generator,
        epochs=20,
        use_multiprocessing=True,
        workers=6)

    def plot_loss_accuracy(history):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history.history["loss"],'r-x', label="Train Loss")
        ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
        ax.legend()
        ax.set_title('cross_entropy loss')
        ax.grid(True)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
        ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
        ax.legend()
        ax.set_title('accuracy')
        ax.grid(True)

        plt.savefig(save_path + "/model_accuracy3")
    
    plot_loss_accuracy(history)

    model.save(save_path + "/ClusteredClassifier3")