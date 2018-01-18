import numpy as np
from numpy import random

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop

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
batch_size = 16


#Pre-process data 
print("Preprocessing Data")
x_train, y_train, x_test, y_test = preprocessing.preprocess(periods, testCategoryNum) 
print("Done.")


batches = batchnumber.getBatches(x_train, batch_size)


model = keras.load('weights_FINAL.h5', custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram, 'Normalization2D':kapre.utils.Normalization2D})

model.summary()


yval = []
ypredict = []

for i in generate.generate(x_train, y_train, batch_size):
    for q in range(i[0].shape[0]):
        x_val = x[0][q]
        y_val = x[1][q]
        yval.append(y_val)
        e = model.predict(x_val)
        ypredict.append(e)

yval = np.array(yval)	
ypredict = np.array(ypredict)	

np.save('yval.npy', yval)
np.save('ypredict.npy', ypredict)

#model.evaluate(x_test, y_test, verbose=1, sample_weight=None, steps=None)
