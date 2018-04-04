# Emopy

Emopy is python module for training emotion recognition from face images models

## Getting Started
This module contains the following types of networks for emotion recognition
### Face Image input model
This network takes face images of shape(batch_size,48,48,1) as input.
#### How to run trainining program
```
python -m train --net face --emotions typeof-emotion-classsification --batch batch-size --epochs epoch-number --steps number-of-steps-per-epoch --lr learning-rate --dataset_path dataset-path --model_path path-to-save-model-without-extension --augmentation use-augmentation --verbose print-logs --sequence_length valid-only-for-rnn-networks
```
Where ```--net`` is type of network to train. For face image input model it is face.
To use other type of network see the following list

* __face__ : Face image input model
* __dlib__ : dlib points features inputs model
* __face+dlib__ : Face image and dlib points inputs model
* __vgg-face__ : Face image inputs model fined tuned from vgg-face architecture.
* __rnn__ : Sequential images input RNN model
* __dlib-rnn__ : Sequential images and dlib points features inputs RNN model

##### ```--emotions```
This module can be used to train both seven basic emotions(ANGER,DISGUST,FEAR,HAPPY,SAD,SURPRISE AND NEUTRAL) classifier,positive neutral emotions(POSITIVE AND NEUTRAL) classifier or positive negetive emotions classifier
values of ```--emotions``` 
* all : for seven emotion classifier
* pos-neu : for positive neutral classifier
* pos-neg : for positive negative classifier

##### Network parameters
* batch : batch size, default 32
* epochs : number of epochs, default 100 
* steps : number of steps per epoch, default 1000
* lr : learning rate , default 1e-4
##### ```--dataset_path```
Path to dataset. For non sequence classifications this directory should contain  train and test folders.
##### ```--model_path```
Path to save model without extesion. e.g /home/user/models/model can be used to save model files json(/home/user/models/model.json) and h5(/home/user/models/model.h5).  other parameters are self explanatory.



