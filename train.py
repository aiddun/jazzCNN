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
import generate

periods = [[1923, 1930], [1931, 1944], [1945, 1949], [1950, 1955]]
testCategoryNum = 4

epochs = 500

#Random Seed
np.random.seed(4)
audio_frequency = 16000
batch_size = 16



#Pre-process data 
print("Preprocessing Data")
x_train, y_train, x_test, y_test = preprocessing.preprocess(periods, testCategoryNum) 
print("Done.")


#Initialize Callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
saveModel = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', monitor='epoch', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=10)



#Define Model
model = Sequential()

model.add(Melspectrogram(input_shape=(1, 240000), sr=16000, n_mels=128, fmin=0.0, fmax=None,
                                                power_melgram=1.0, return_decibel_melgram=True,
                                                trainable_fb=True, trainable_kernel=True))

model.add(Normalization2D(int_axis=0))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


#Batch Size Calculatons
batches = int((x_train.size) / batch_size)
print(str(batches) + " batches, with a batch size of " + str(batch_size) + ".")
extraBatchSize = (x_train.size) % batches


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )

print("")

model.summary()


generate.generate(x_train, y_train, batches, batch_size, extraBatchSize)
                   
model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=1, callbacks=[tbCallBack, saveModel])


model.save('weights_FINAL.h5')

#model.evaluate(x_test, y_test, verbose=1, sample_weight=None, steps=None)
