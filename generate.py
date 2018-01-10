import numpy as np
from numpy import random
import scipy.io.wavfile

def train(x_batch, y_batch):
    x_batch_final = []
    y_batch_final = []

    for c in range(x_batch.size):
        x_batch_batch = x_batch
        y_batch_batch = y_batch

        print("nextfile")
        x_rate, x_audio = scipy.io.wavfile.read(x_batch_batch)
        x_audio = x_audio[x_rate * 300: -(x_rate * 300)]

        audiosize = audio-frequency * 15
        cuts = int(x_audio.size / (audiosize))
        x_final = np.ndarray((cuts, 1, audiosize), dtype=int)
        y_final = np.ndarray((cuts), dtype=int)
        
        for d in range(cuts):
            loweraud = int(d * audiosize)
            upperaud = int(((d+1) * audiosize))
            x_data = x_audio[slice(loweraud, upperaud)]
            x_data = np.array(x_data)
            x_data = np.reshape(x_data, (1, audiosize))
            x_final[d] = x_data
            y_final[d] = y_batch_batch

        x_batch_final.append(x_final)
        y_batch_final.append(y_final)

    x_batch_final = np.concatenate((x_batch_final[:]))
    y_batch_final = np.concatenate((y_batch_final[:]))
    
    #Randomly shuffles sub-batch to prevent
    perm = np.random.permutation(x_batch_final.shape[0])
    x_batch_final = x_batch_final[perm]
    y_batch_final = y_batch_final[perm]

    print(y_batch_final.shape)
    print(x_batch_final.shape)

    return (x_batch_final, y_batch_final)

def generate(x_train, y_train, batches, batch_size, extraBatchSize):

    x_batch = x_train[int(y_train.size/2):]
    y_batch = y_train[int(y_train.size/2):]

    inputs, targets = train(x_batch, y_batch)
    return inputs, targets

    #Extra batch
    x_batch = x_train[-int(y_train.size/2):]
    y_batch = y_train[-int(y_train.size/2):]
    return train(x_batch, y_batch)
