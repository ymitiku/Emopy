from keras.layers  import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from config import LEARNING_RATE, EPOCHS,BATCH_SIZE,DATA_SET_DIR
from preprocess.dataset_process import load_dataset
import os
import keras
import numpy as np
from loggers.base import EmopyLogger


class NeuralNet(object):
    """
    Base class for all neural nets.

    Parameters
    ----------
    input_shape : tuple
    
    """
    
    def __init__(self,input_shape,logger=None):
        self.input_shape = input_shape
        self.model = self.build()
        if logger is None:
            self.logger = EmopyLogger()
        else:
            self.logger = logger
    def build(self):
        """
        Build neural network model
        
        Parameters
        ----------
        
        Returns 
        -------
        keras.models.Model : 
            neural network model
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding= "valid", input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu',padding= "valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu',padding= "valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu',padding= "valid"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='softmax'))
        self.feature_extractors = ["image"]
        self.built = True
        return model
    def save_model(self,model):
        pass
    def train(self):
        x_train, y_train = load_dataset(os.path.join(DATA_SET_DIR,"train"),True)
        x_test , y_test  = load_dataset(os.path.join(DATA_SET_DIR,"test"),True)

        image_shape = (-1 , self.input_shape[0], self.input_shape[1], self.input_shape[2])
        print(image_shape)
        x_train = x_train.reshape(image_shape)
        x_test = x_test.reshape(image_shape)


        y_train = np.eye(7)[y_train]
        y_test = np.eye(7)[y_test]
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(LEARNING_RATE),
                    metrics=['accuracy'])
        self.model.fit(x_train,y_train,epochs = EPOCHS, 
                        batch_size = BATCH_SIZE,validation_data=(x_test,y_test))


    def test(self):
        pass
