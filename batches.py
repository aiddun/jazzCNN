import os
import numpy as np

rate = 16000

def getBatches(x_train):
    batches = 0
    for u in x_train:
        batches += batchGet(u)
    print(batches + " batches.")
    return batches


def batchGet(path):

    audio = os.path.getsize(path)

    sampletime = 5
    samplesize = sampletime * x_rate

    audio_padding_mins = 60 * 5

    audio = audio - 2(audio_padding_mins * rate)

    pathBatches = audio // samplesize
    return pathBatches