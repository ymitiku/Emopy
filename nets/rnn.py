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


class LSTMNet(NeuralNet):
    def __init__(self,input_shape,convnet_model_path=None,preprocessor = None,logger=None,train=True):
        self.convnet_model_path = convnet_model_path;
        self.max_sequence_length = 64
        NeuralNet.__init__(self,input_shape,preprocessor,logger,train)
        # self.X,self.y = self.preprocessor.load_sequencial_dataset("dataset/ck-sequence",max_sequence_length  = self.max_sequence_length)
        # self.X,self.y = shuffle(self.X,self.y)
        # self.X = self.X.astype(np.float32)/255.0;
        # self.y = self.y.reshape(-1,7);
        # self.X = self.X.reshape(-1,self.max_sequence_length,self.input_shape[0]*self.input_shape[1]*self.input_shape[2])
        self.model = self.build()
        self.models_local_folder = "rnn"
    def build(self):
        # convnet_model = self.load_model()
        # for layer in convnet_model.layers:
        #     layer.trainable = False

        # convnet_model.layers.pop()
        # convnet_model.layers.pop()
        # print "conv net",convnet_model.summary()
        # convnet_model.summary()

        # input_layer = Input()

        # # # conv_input = Reshape((self.input_shape))(input_layer)
        # # # conv_input = K.reshape(input_layer,(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
        # conv_input = Lambda(lambda x: K.reshape(x,(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2])))(input_layer)

        # convnet_model = convnet_model(conv_input)
        # # last_layer = convnet_model.layers[12].output
        # # last_layer_shape = last_layer.shape

        # # # last_layer = K.reshape(last_layer,(-1,last_layer_shape[1],self.max_sequence_length,1))
        # convnet_model = Lambda(lambda x: K.reshape(x,(-1,self.max_sequence_length,1)))(convnet_model)
        
        
        # lstm_input_layer = Reshape((self.max_sequence_length,self.input_shape[0]*self.input_shape[1]*self.input_shape[2]))(input_layer)
        # # print self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        # lstm_model = LSTM(64,input_shape=(self.max_sequence_length,(self.input_shape[0]*self.input_shape[1]*self.input_shape[2])), return_sequences=False)
        # lstm_model = lstm_model(lstm_input_layer)
        # print "lstm _model",lstm_model.shape
        # lstm_model = Lambda(lambda x: K.reshape(x,(-1,self.max_sequence_length,1)))(lstm_model)
        # print "convNet",convnet_model.shape
        # print "lstm",lstm_model.shape
        # help(merge)
        # x = merge([convnet_model,lstm_model],mode="dot",dot_axes=-2)
        # x = multiply([convnet_model,lstm_model])

        # predictions = Dense(self.preprocessor.classifier.get_num_class(),activation="softmax")(lstm_model)

        # model = Model(inputs=input_layer,outputs=predictions)

        # model = Sequential()
        # model.add(Conv2D(32,(5,5),input_shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2]),padding="valid",strides=1))
        # conv_input_shape = self.max_sequence_length, self.input_shape[0],self.input_shape[1],self.input_shape[2]
        # model.add(TimeDistributed(Conv2D(64, (3, 3)),
                        #   input_shape=(conv_input_shape)))
        # model.add(Bidirectional(LSTM(64, return_sequences=True)))
        # model.add(Bidirectional(LSTM(128,return_sequences=False)))
        # model.add(TimeDistributed(Dense(6)))
        # model.add(Activation('softmax'))

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='valid', activation='relu'),input_shape=(self.max_sequence_length, 48, 48, 1)))
        # model.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='valid', activation='relu')))
        model.add(TimeDistributed(Dropout(0.2)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(252,activation="relu")))
        model.add(TimeDistributed(Dropout(0.2)))
        # model.add(LSTM(64,return_sequences=True,stateful=False))
        model.add(LSTM(64,return_sequences=False,stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(128,activation="softmax"))
        model.add(Dropout(0.2))
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
        # x_train,x_test,y_train ,y_test = train_test_split(self.X,self.y,test_size=0.3)
        # self.model.fit(x_train,y_train,epochs = EPOCHS, 
        #                 batch_size = BATCH_SIZE,validation_data=(x_test,y_test))
        self.model.fit_generator(self.preprocessor.flow(),steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=(self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels))
        
        score = self.model.evaluate(self.preprocessor.test_images, self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.models_local_folder, score)
    