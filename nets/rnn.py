from nets.base import NeuralNet
from keras.models import model_from_json,Model,Sequential
from keras.layers import Input,LSTM,Reshape,GlobalAveragePooling2D,Dense,Lambda,Conv2D,Bidirectional,Activation,TimeDistributed
import os
import cv2
import dlib
import numpy as np
from train_config import BATCH_SIZE,LEARNING_RATE,EPOCHS,STEPS_PER_EPOCH
from sklearn.model_selection import train_test_split
import keras
from keras.layers import multiply,merge,Conv1D,MaxPooling1D,MaxPooling2D,Flatten,Dropout
from keras import backend as K
from keras.layers.embeddings import Embedding
from sklearn.utils import shuffle
from train_config import LOG_DIR
from loggers.base import EmopyLogger


class LSTMNet(NeuralNet):
    def __init__(self,input_shape,convnet_model_path=None,preprocessor = None,logger=None,train=True):
        self.convnet_model_path = convnet_model_path;
        self.max_sequence_length = 6
        NeuralNet.__init__(self,input_shape,preprocessor,logger,train)
        self.models_local_folder = "rnn"
        self.logs_local_folder = self.models_local_folder
        if not os.path.exists(os.path.join(LOG_DIR,self.logs_local_folder)):
            os.makedirs(os.path.join(LOG_DIR,self.logs_local_folder))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt")])
        else:
            self.logger = logger
        self.model = self.build()
    def build(self):
       

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='valid', activation='relu'),input_shape=(self.max_sequence_length, 48, 48, 1)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', activation='relu')))
        # model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(128,return_sequences=False,stateful=False,dropout=0.4))
        model.add(Dense(6,activation="softmax"))
        
        return model;
        
    def load_model(self):
        if self.convnet_model_path==None:
            self.convnet_model_path = "models/nn/nn-5"
        with open(self.convnet_model_path+".json") as model_file:
            model = model_from_json(model_file.read())
            model.load_weights(self.convnet_model_path+".h5")
            return model
    def train(self):
        """Traines the neuralnet model.      
        This method requires the following two directory to exist
        /PATH-TO-DATASET-DIR/train
        /PATH-TO-DATASET-DIR/test
        
        """
        
        print "model"
        self.model.summary()
        print "learning rate",LEARNING_RATE
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(LEARNING_RATE),
                    metrics=['accuracy'])
        print self.model.output.shape

        self.model.fit_generator(self.preprocessor.flow(),steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=(self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels))
        
        score = self.model.evaluate(self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels)
        self.save_model()
        self.logger.log_model(self.models_local_folder, score)
    def predict(self,sequence_faces):
        assert sequence_faces[0].shape == IMG_SIZE, "Face image size should be "+str(IMG_SIZE)
        face = face.reshape(-1,self.max_sequence_length,48,48,1)
        emotions = self.model.predict(face)[0]
        return emotions
    