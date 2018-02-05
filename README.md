# jazzCNN
 
An experiment looking at model confidence of a trained classifier as a potential metric of quantitative chronological Jazz style progression using a deep CNN.

**Blog post:** (coming soon)

3 second 16000hz audio files were converted to Mel Spectrograms on the fly with the GPU using Kapre, and then fed into a deep CNN trained as a classifier. My main intention was to see if the mean Softmax confidence per category of each category could be looked at as a metric of similarity/progression between periods.

Trained on a dataset scraped from Internet Archive's [The David W. Niven Collection of Early Jazz Legends](https://archive.org/details/davidwnivenjazz). Samples were converted into 16000hz wavform with the first and last 5 minutes of each track cut to remove some track commentary. Track list attached. Note some entries have multiple tapes. The dataset came out to be over 60GB, so I unfortunately can't host it, but I still have it in an S3 bucket.

The time periods I used were Early Jazz (1920-1930), Swing/Big Band (1931-1944), Bop (1945-1959), and Cool Jazz (1950-1955).

  
## Requirements
- Python 3
- Tensorflow
- Keras 2 (and dependencies)
- [Kapre](https://github.com/keunwoochoi/kapre)
- Numpy
- Scipy
  
  
  

## Network Structure
I realise it's easier to just paste the network code than to have a giant network summary table. Also available in train.py. Input is a three second numpy wavfile array. 

```
model = Sequential()

model.add(Melspectrogram(input_shape=(1, 48000), sr=32, n_mels=64, fmin=0.0, fmax=None,
                                                power_melgram=1.0, return_decibel_melgram=True,
                                                trainable_fb=True, trainable_kernel=True))

model.add(Normalization2D(int_axis=-1))    

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

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

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax'))



opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt)

```

## Usage

The program will look for 16000hz wavform files in ./FINAL/19**/. I still need to add console keyword arguments to tune hyperparameters. To train, run train.py.


## Results and Conclusions

Results were pretty interesting, to an extent, and offer a clever premice, but require much future work. Loss began to rise and accuracy began to drop after continuous epochs, but the results for the first epoch were pretty promising, however with a potential possibility of severe underfitting, even with 347,990 training samples. Also, the dataset needs further preprocessing cleaning as there still exists some of Niven's commentary on tracks, which removal could potentially be automated. The network structure could be modified, and the *chronological similarity of the model accuracy over time* demonstrates and warrents a need for further testing and revisiting. Unfortunately, I can only afford so many EC2 GPU hours at the moment.
In a revisit, it may actually be wiser to train "yes or no" 4 binary classifiers, one for each period, and then feed all of the samples through those in order to more effectively train and recognize features.

Future things to do with data:
- Feature visualization
- Style transfer?

#### Results:

|   | **Early Jazz (1920-1930)** |**Swing/Big Band (1931-1944)**|**Bop (1945-1959)**|**Cool Jazz (1950-1955)**|
| --- | --- | --- | --- | --- |
| **Early Jazz (1920-1930)** | **0.371** | 0.205 | 0.190 | 0.152 |
| **Swing/Big Band (1931-1944)**| 0.205 | **0.200** | 0.322 | 0.165 |
| **Bop (1945-1959)**| 0.190 | 0.322 |**0.538** | 0.324 |
| **Cool Jazz (1950-1955)**| 0.152 | 0.165 | 0.324 |**0.173** |

**Graphs:** Coming soon.
