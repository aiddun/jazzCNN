# jazzCNN
 
An experiment looking at mean confidence of a unmodified deep convolutional classifier as a potential metric of quantitative chronological Jazz style progression using a deep CNN. My premise with this project was if a model, even a simple deep convolutional classifier, can express its Softmax confidence that a sample of audio possesses certain common identified "stylistic" features, then using that confidence over an extremely large population of samples could potentially yield a basic rudimentary metric of similarity between categories.

**Blog post:** (coming soon)

I chose to use Jazz music in this project because of the district opportunities it offers with its unique history, evolution, and context. Jazz can be argued to be one of the most important creative forms of human expression in US history, where musicians often use improvisation to create new music extemporaneously. While much is improvised, historically, Jazz musicians have often played in the style of performing of the time, with trends of the music during of the time period dictating the decisions that musicians made. The history of Jazz is rich and complex, constantly and rapidly evolving and changing in complex ways, particularly present in the early eras of Jazz. As a trumpet player myself, I have always been fascinated by Jazz. The ability to have a deep comprehension of music theory, especially of Jazz, takes many years to develop, with understand Jazz history as a whole, requiring many years of academic study. The goal of this project was to see if this could also be observed by a computer, and if it could provide a metric of similarity between period.

I owe a special thanks to the Keras Audio Preprocessor library, or "Kapre" by Keunwoo Choi, which allowed for a seamless conversion of wavform audio samples to two-dimensional Mel series Spectrographs, allowing a normal 2D Convnet to train on the dataset while emulating human audio perception. I was rushed for time for many parts of this project so I apologize that my code isn't the as formidable as it should be.

Kapre is unique in that it is able to compute Spectrograms on the fly using the GPU, which greatly motivated my choice to just use Keras with a Tensorflow backend in writing the network graph, with being cut for time another factor. Plus, the goal of this project was to determine if a standard "feed-forward" convnet could compute a style metric without modified structure. 3 second 16000hz audio files were converted to Mel Spectrograms, and then fed into a deep CNN trained as a classifier. My main intention was to see if the mean Softmax confidence per category of each category could be used at as a simple way of obtaining a metric of similarity/progression between periods. 

The network was trained on a dataset scraped, sorted, and transcoded from Internet Archive's [The David W. Niven Collection of Early Jazz Legends](https://archive.org/details/davidwnivenjazz). Samples were converted into 16000hz waveform with the first and last 5 minutes of each track cut to remove some of Niven's track commentary. Track list attached. Note some entries have multiple tapes. The dataset came out to be over 60GB, so I unfortunately can't host it, but I still have it and its tarball in an S3 bucket.

The historical time periods I used were Early Jazz (1920-1930), Swing/Big Band (1931-1944), Bop (1945-1959), and Cool Jazz (1950-1955).

### Sample Mel Spectrogram
  
![alt text](https://raw.githubusercontent.com/AidDun/jazzCNN/master/img/melspec.png "Sample Melspectrogram")
  
  
## Requirements
- Python 3
- Tensorflow
- Keras 2 (and dependencies)
- [Kapre](https://github.com/keunwoochoi/kapre)
- Numpy
- Scipy
  
  
  

## Network Structure
It's easier to just paste the network code than to have a giant network summary table. Also available in train.py. Input is a three second numpy wavfile array. The network came out to 2,482,868 parameters total. I used the RMSProp optimiser with a learning rate of 0.0001. 

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

The program will look for 16000hz wavform files in ./FINAL/19\*\*/. I still need to add console keyword arguments to tune hyperparameters. To train, run \__train/train.py. To iterate trained weights over the dataset, use \_evaluate/confidence_calculate.py. I've uploaded the first epoch training weights to the repo. The pretrained weights use custom layers in Keras, so the following structure will be needed to load the weights:
```
model = load_model('weights.00.hdf5', custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram, 'Normalization2D':kapre.utils.Normalization2D})
```


## Results and Conclusions

Results were pretty interesting, to an extent, and offer a clever premise, but require much future research. Loss began to rise and accuracy began to drop after continuous epochs, while the trained weights extracted from the first epoch produced interesting results. However, there exits a large possibility of the results not meaning anything, as the upwards climbing loss on a Tensorboard graph shows. However, not all experiments are successful, and I would be fine with insignifcant results.

The model was able to "identify" a negative trend in similarity from early Jazz as time progresses, as well as producing an interesting nonlinear progression from previous period graph. The use of one epoch also has a potential of being justified. Of course, any quantitative metric of similarity at all between forms or art should be taken with a large grain of salt, but the results produced were pretty interesting. Using solely mean Softmax was an interesting premise, but there might definitely be better ways of doing this as well.

In the future, the dataset needs further preprocessing and scrubbing as there still exists some of Niven's commentary on tracks, which removal could potentially be automated. Furthermore, hyperparameters such as the learning rate could be heavely tuned. There may be a clear error in plain sight in the parameters that just passed by me. The network structure could be modified, as the *chronological similarity of the model accuracy over time* demonstrates and warrents a need for further testing and revisiting. Unfortunately, I can only use so many EC2 GPU hours at the current moment. Through this project, I've found that the Internet Archive is an incredible resource for machine learning data, containing many usually uncommon compiled groups of media pre-labeled in many cases.

In a revisit, it may be interesting to see if it would be wiser to  train 4  binary classifiers, one for each period, and then feed all of the samples through those in order to more effectively train and identify features as more memory and neurons can be devoted to feature. As there are almost an infinite amount Compiling the dataset took *much* longer than I expected, but this entire project was an incredible experiance in almost every area. 

Future things to do with data:
- Feature visualization
- Style transfer?

### Results:

|   | **Early Jazz (1920-1930)** |**Swing/Big Band (1931-1944)**|**Bop (1945-1959)**|**Cool Jazz (1950-1955)**|
| --- | --- | --- | --- | --- |
| **Early Jazz (1920-1930)** | **0.371** | 0.205 | 0.190 | 0.152 |
| **Swing/Big Band (1931-1944)**| 0.205 | **0.200** | 0.322 | 0.165 |
| **Bop (1945-1959)**| 0.190 | 0.322 |**0.538** | 0.324 |
| **Cool Jazz (1950-1955)**| 0.152 | 0.165 | 0.324 |**0.173** |
  
  
  
### Graphs: 
Model Progression over Time (Source: Tensorboard)
  

<img src="https://raw.githubusercontent.com/AidDun/jazzCNN/master/img/graphs.PNG" width=625>
  
Period Deveation from First Period over time
  
  
<img src="https://raw.githubusercontent.com/AidDun/jazzCNN/master/img/deviation.PNG" width=750>

Deveation from Previous Period over time compensated for correct value (results/confidence for true value)
 
  
<img src="https://raw.githubusercontent.com/AidDun/jazzCNN/master/img/priorperiod.PNG" width=625>

---

Questions or comments about the project? Please reach out to me! I would really appreciate feedback, as this was the first time I attemped such a large scale ML project of this caliber. Feel free to raise an issue and I would love to talk.
