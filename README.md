# jazzCNN
A deep convolutional neural network for the classification of Jazz for use in style comparison and future use for Style Extraction/Transfer built in Keras. 15 second 16000hz audio files are converted to Melspectrograms with the GPU on the fly using kapre, and are then fed into a deep CNN. 

For training on a dataset scraped from [The David W. Niven Collection of Early Jazz Legends](https://archive.org/details/davidwnivenjazz). Complete track list attached. Note some entries have multiple tapes.

  
  

![Alt text](img/melspec.png?raw=true "Sample Melspectrogram")
Sample Melspectrogram


Layer (type)                 | Output Shape             | Param #
| --- | --- | --- |
melspectrogram-1 (Melspectrogram) | (None, 128, 938, 1)      |   296064   
normalization2d-1 (Normalization)  | (None, 128, 938, 1)      |   0      
conv2d-1 (Conv2D)            | (None, 128, 938, 32)     |   320
activation-1 (Activation)    | (None, 128, 938, 32)     |   0
conv2d-2 (Conv2D)            | (None, 126, 936, 32)     |   9248
activation-2 (Activation)    | (None, 126, 936, 32)     |  0
max-pooling2d-1 (MaxPooling2D)   | (None, 63, 468, 32)      |   0
dropout-1 (Dropout)          | (None, 63, 468, 32)      |   0
conv2d-3 (Conv2D)            | (None, 63, 468, 64)      |   18496
activation-3 (Activation)    | (None, 63, 468, 64)      |   0
conv2d-4 (Conv2D)            | (None, 61, 466, 64)      |   36928
activation-4 (Activation)    | (None, 61, 466, 64)      |   0
max-pooling2d-2 (MaxPooling2D)   | (None, 30, 233, 64)      |   0
dropout-2 (Dropout)          | (None, 30, 233, 64)      |   0
conv2d-5 (Conv2D)            | (None, 30, 233, 128)     |   73856
activation-5 (Activation)    | (None, 30, 233, 128)     |   0
conv2d-6 (Conv2D)            | (None, 28, 231, 128)     |   147584
activation-6 (Activation)    | (None, 28, 231, 128)     |   0
max-pooling2d-3 (MaxPooling2D) | (None, 14, 115, 128)     |   0
dropout-3 (Dropout)          | (None, 14, 115, 128)     |   0
conv2d-7 (Conv2D)            | (None, 14, 115, 256)     |   295168
activation-7 (Activation)    | (None, 14, 115, 256)     |   0
conv2d-8 (Conv2D)            | (None, 12, 113, 256)     |   590080
activation-8 (Activation)    | (None, 12, 113, 256)     |   0
max-pooling2d-4 (MaxPooling2D) | (None, 6, 56, 256)       |   0
dropout-4 (Dropout)          | (None, 6, 56, 256)       |   0
flatten-1 (Flatten)          | (None, 86016)            |   0
dense-1 (Dense)              | (None, 32)               |   2752544
dropout-5 (Dropout)          | (None, 32)               |   0
dense-2 (Dense)              | (None, 512)              |   16896
dropout-6 (Dropout)          | (None, 512)              |   0
dense-3 (Dense)              | (None, 512)              |   262656
dropout-7 (Dropout)          | (None, 512)              |   0
dense-4 (Dense)              | (None, 512)              |   262656
dropout-8 (Dropout)          | (None, 512)              |   0
dense-5 (Dense)              | (None, 4)                |   2052
Total params: 4,764,548 
Trainable params: 4,764,548
Non-trainable params: 0

_________________________________________________________________
