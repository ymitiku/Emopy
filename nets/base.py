from keras.layers  import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from config import LEARNING_RATE, EPOCHS,BATCH_SIZE,DATA_SET_DIR,LOG_DIR
from config import PATH2SAVE_MODELS
from preprocess.dataset_process import load_dataset
import os
import keras
import numpy as np
from loggers.base import EmopyLogger
from constants import EMOTIONS
import time


class NeuralNet(object):
    """
    Base class for all neural nets.

    Parameters
    ----------
    input_shape : tuple
    
    """
    
    def __init__(self,input_shape,logger=None):
        self.input_shape = input_shape
        self.models_local_folder = "nn"
        self.logs_local_folder = self.models_local_folder
        if logger is None:
            if not os.path.exists(os.path.join(LOG_DIR,self.logs_local_folder)):
                os.makedirs(os.path.join(LOG_DIR,self.logs_local_folder))
            self.logger = EmopyLogger([os.path.join(LOG_DIR,self.logs_local_folder,"nn.txt")])
        else:
            self.logger = logger
        self.feature_extractors = ["image"]
        self.number_of_class = len(EMOTIONS)
        self.model = self.build()
        
    def build(self):
        """
        Build neural network model
        
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
        model.add(Dense(self.number_of_class, activation='softmax'))
        
        self.built = True
        return model
    def log_model(self,score):
        model_number = np.fromfile(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,"model_number.txt"),dtype=int)
        model_file_name = self.models_local_folder+"-"+str(model_number[0]-1)
    
        self.logger.log("**************************************")
        self.logger.log("Trained model "+model_file_name+".json")
        self.logger.log(time.strftime("%A %B %d,%Y %I:%M%p"))
        self.logger.log("Dataset dir: "+DATA_SET_DIR)
        self.logger.log("Parameters")
        self.logger.log("_______________________________________")
        self.logger.log("Batch-Size    : "+str(BATCH_SIZE))
        self.logger.log("Epoches       : "+str(EPOCHS))
        self.logger.log("Learning rate : "+str(LEARNING_RATE))
        self.logger.log("_______________________________________")
        self.logger.log("Loss          : "+str(score[0]))
        self.logger.log("Accuracy      : "+str(score[1]))
        self.logger.log("**************************************")
        
    def save_model(self):
        """
        Saves NeuralNet model. The naming convention is for json and h5 files is,
        `/path-to-models/model-local-folder-model-number.json` and  
        `/path-to-models/model-local-folder-model-number.h5` respectively.
        This method also increments model_number inside "model_number.txt" file.
        """
        
        if not os.path.exists(PATH2SAVE_MODELS):
            os.makedirs(PATH2SAVE_MODELS)
        if not os.path.exists(os.path.join(PATH2SAVE_MODELS,self.models_local_folder)):
            os.makedirs(os.path.join(PATH2SAVE_MODELS,self.models_local_folder))
        if not os.path.exists(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,"model_number.txt")):
            model_number = np.array([0])
        else:
            model_number = np.fromfile(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,"model_number.txt"),dtype=int)
        model_file_name = self.models_local_folder+"-"+str(model_number[0])
        with open(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,model_file_name+".json"),"a+") as jfile:
            jfile.write(self.model.to_json())
        self.model.save_weights(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,model_file_name+".h5"))
        model_number[0]+=1
        model_number.tofile(os.path.join(PATH2SAVE_MODELS,self.models_local_folder,"model_number.txt"))

    def train(self):
        """Traines the neuralnet model.      
        This method requires the following two directory to exist
        /PATH-TO-DATASET-DIR/train
        /PATH-TO-DATASET-DIR/test
        
        """
        assert os.path.exists(os.path.join(DATA_SET_DIR,"train")), "Training dataset path :"+os.path.join(DATA_SET_DIR,"train")+", doesnot exist." 
        assert os.path.exists(os.path.join(DATA_SET_DIR,"test")), "Test dataset path :"+os.path.join(DATA_SET_DIR,"test")+", doesnot exist." 
        
        x_train, y_train = load_dataset(os.path.join(DATA_SET_DIR,"train"),True)
        x_test , y_test  = load_dataset(os.path.join(DATA_SET_DIR,"test"),True)

        image_shape = (-1 , self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        x_train = x_train.reshape(image_shape)
        x_test = x_test.reshape(image_shape)


        y_train = np.eye(self.number_of_class)[y_train]
        y_test = np.eye(self.number_of_class)[y_test]
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(LEARNING_RATE),
                    metrics=['accuracy'])
        self.model.fit(x_train,y_train,epochs = EPOCHS, 
                        batch_size = BATCH_SIZE,validation_data=(x_test,y_test))
        score = self.model.evaluate(x_test,y_test)
        self.save_model()
        self.log_model(score)

    def test(self):
        pass
