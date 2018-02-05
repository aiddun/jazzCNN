import numpy as np
from numpy import random
import scipy.io.wavfile


def trainGet(x_batch_batch):
    print("nextfile")

    sampletime = 3
    x_rate, x_audio = scipy.io.wavfile.read(x_batch_batch)
    samplesize = sampletime * x_rate
    audio_padding_mins = 60 * 5
    x_audio = x_audio[x_rate*300 : -(x_rate*300 + x_audio.size%samplesize)]

    audioNum = int(x_audio.size / (sampletime * x_rate))
    x_audio = np.array(np.array_split(x_audio, audioNum))
    return np.reshape(x_audio, (x_audio.shape[0], 1, x_audio.shape[1]))



def generate(x_train, y_train, batch_size):
    #Just need to split array into batches of 4
    # 
    g = 0
    print("gen called")
    for i in range(len(x_train)):
        x_yield = trainGet(x_train[i])
        y_yield = y_train[i]
        print(g)
        g += 1

        for j in x_yield:
            j = np.reshape(j, (1, 1, 48000))
            yield j, y_yield
