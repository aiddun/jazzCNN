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
import generate1

periods = [[1923, 1930], [1931, 1944], [1945, 1949], [1950, 1955]]
testCategoryNum = 4

epochs = 5

#Random Seed
np.random.seed(4)
audio_frequency = 16000
batch_size = 1


#Pre-process data 
print("Preprocessing Data")
x_train, y_train, x_test, y_test = preprocessing.preprocess(periods, testCategoryNum) 
print("Done.")


batches = batchnumber.getBatches(x_train, batch_size)


model = load_model('./weights_FINAL.h5', custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram, 'Normalization2D':kapre.utils.Normalization2D})

model.summary()


yval = [[],[],[],[]]

for i in generate1.generate(x_train, y_train, batch_size):
        e = model.predict(i[0])
        yval[(i[1])].append(e)

yval = np.array(yval)	

np.save('yval.npy', yval)

#model.evaluate(x_test, y_test, verbose=1, sample_weight=None, steps=None)
