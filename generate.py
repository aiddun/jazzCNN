import numpy as np
from numpy import random
import scipy.io.wavfile
import scipy.signal

def train(x_batch, y_batch):
    x_batch_final = np.ndarray((0, 1, 32000), dtype="int16")
    y_batch_final = np.ndarray((0), dtype=int)
    
    for h in range(x_batch.shape[0]):
        x_batch_final = np.concatenate( (x_batch_final, trainGet(x_batch[h])) )
        y_batch_diff = x_batch_final.shape[0] - y_batch_final.shape[0]
        y_batch_add = np.full((y_batch_diff), y_batch[h])
        y_batch_final = np.concatenate( (y_batch_final, y_batch_add) )


    randomize = np.random.permutation(x_batch_final.shape[0])
    x_batch_final = x_batch_final[randomize]
    y_batch_final = y_batch_final[randomize]

    return (x_batch_final, y_batch_final)


def trainGet(x_batch_batch):
    print("nextfile")

    sampletime = 2
    x_rate, x_audio = scipy.io.wavfile.read(x_batch_batch)
    print("File loaded.")
    samplesize = sampletime * x_rate
    audio_padding_mins = 60 * 5
    x_audio = x_audio[x_rate*300 : -(x_rate*300 + x_audio.size%samplesize)]

    audioNum = int(x_audio.size / (sampletime * x_rate))
    x_audio = np.array(np.array_split(x_audio, audioNum))
    return np.reshape(x_audio, (x_audio.shape[0], 1, x_audio.shape[1]))



def generate(x_train, y_train, batch_size):
    #Just need to split array into batches of 4
    # 
    print("gen called")
    print(x_train.shape)
    batches = round(y_train.size/batch_size)
    x_train = np.split(np.array(x_train), batches)
    y_train = np.split(np.array(y_train), batches)

    while 1:
        for x in range(batches):
            inputs, targets = train(x_train[x], y_train[x])
            print(inputs.shape)
            batchbatches = round(targets.size/batch_size)
            final_inputs = np.array_split(np.array(inputs), batchbatches)
            final_target = np.array_split(np.array(targets), batchbatches)
            final_inputs = np.array(final_inputs)
            print(final_inputs.shape[0])
            for i in range(final_inputs.shape[0]):
                yield final_inputs[i], final_target[i]
