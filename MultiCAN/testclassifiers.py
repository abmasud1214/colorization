import clusterclassifier
import tensorflow as tf
from tensorflow.keras.models import Model

import numpy as np

from os import listdir

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

data = []
labels = []

path = '../clustereddata/test'

for i, group_folder in enumerate(listdir(path + "/")):
    for filename in listdir(path + "/" + group_folder):
        if filename.endswith('.npy'):
            d = np.load(path + "/" + group_folder + "/" + filename)
            data.append(d)
            labels.append(i)

data_arr = np.array(data)
data_arr = (data_arr - 127.5) / 127.5

classifier = tf.keras.models.load_model("./Models/ClusteredClassifier")

classifier.summary()

predictions = classifier.predict(data_arr)
indices = [np.argmax(x) for x in predictions]
image_category = np.array(indices)

print(labels)
print(image_category)

relative_scores = dict()
relative_scores[0] = {0:0, 1:0, 2:0, 3:0, 4:0}
relative_scores[1] = {0:0, 1:0, 2:0, 3:0, 4:0}
relative_scores[2] = {0:0, 1:0, 2:0, 3:0, 4:0}
relative_scores[3] = {0:0, 1:0, 2:0, 3:0, 4:0}
relative_scores[4] = {0:0, 1:0, 2:0, 3:0, 4:0}

score = 0
for i in range(len(labels)):
    relative_scores[labels[i]][image_category[i]] += 1 
    if labels[i] == image_category[i]:
        score += 1

print(score / len(labels))
print(relative_scores)
print(len(labels))

totals = dict()

for key in relative_scores:
    total = 0
    for i in relative_scores[key]:
        total += relative_scores[key][i]
    totals[key] = total

second_best = {0:0, 1:1, 2:2, 3:3, 4:4}

percent_scores = relative_scores.copy()
for key in percent_scores:
    best = 0
    l = 0
    for i in percent_scores[key]:
        ps = percent_scores[key][i] / totals[key]
        percent_scores[key][i] = ps
        if i != key and ps > best:
            l = i
            best = ps
    second_best[key] = l

print(percent_scores)
print(second_best)

score = 0
for i in range(len(labels)):
    if labels[i] == image_category[i] or image_category[i] == second_best[labels[i]]:
        score += 1

print(score / len(labels))


            