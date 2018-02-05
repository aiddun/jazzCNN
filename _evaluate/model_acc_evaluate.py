import numpy as np
from numpy import random

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from keras.models import load_model

import kapre
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

import preprocessing
import batchnumber
import generate

periods = [[1923, 1930], [1931, 1944], [1945, 1949], [1950, 1955]]
testCategoryNum = 8

epochs = 10

#Random Seed
np.random.seed(4)
audio_frequency = 16000
batch_size = 1


#Pre-process data 
print("Preprocessing Data")
x_train, y_train, x_test, y_test = preprocessing.preprocess(periods, testCategoryNum) 
print("Done.")

print(y_test)

batches = batchnumber.getBatches(x_train, batch_size)

exit()


model = load_model('./weights_final.h5', custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram, 'Normalization2D':kapre.utils.Normalization2D})

print("x_test", x_test.shape)
print("x_train", x_train.shape)

model.evaluate_generator(generate.generate(x_test, y_test, batch_size), steps=batches)
