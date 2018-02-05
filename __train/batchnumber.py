import os
import numpy as np

rate = 16000

def getBatches(x_train, batch_size):
    batches = 0

    for u in x_train:
        batches += batchGet(u)

    print(str(batches) + " batches.")
    print("BATCHES", batches)

    return batches


def batchGet(path):

    audio = os.path.getsize(path)

    sampletime = 3
    samplesize = sampletime * rate

    audio_padding_mins = 60 * 5

    audio = audio - (2 * (audio_padding_mins * rate))

    pathBatches = audio // samplesize
    return pathBatches
