from keras.layers  import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model, model_from_json
from train_config import LEARNING_RATE,EPOCHS,BATCH_SIZE,DATA_SET_DIR,LOG_DIR,STEPS_PER_EPOCH
from test_config import MODEL_PATH
import os
import keras
import numpy as np
from loggers.base import EmopyLogger
import time
from util import SevenEmotionsClassifier
from config import IMG_SIZE
from nets.base import NeuralNet




class MultiInputNeuralNet(NeuralNet):
    """
    Neutral network whose inputs are images, dlib points, dlib points distances from centroid point
    and dlib points vector angle with respect to centroid vector.

    Parameters
    ----------
    input_shape : tuple
    
    """
    
    def __init__(self,input_shape,preprocessor = None,logger=None,train=True):
        self.input_shape = input_shape
        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (48,48,1)" 
        self.models_local_folder = "minn"
        self.logs_local_folder = self.models_local_folder
        self.preprocessor = preprocessor
        
        if not os.path.exists(os.path.join(LOG_DIR,self.logs_local_folder)):
            os.makedirs(os.path.join(LOG_DIR,self.logs_local_folder))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt")])
        else:
            self.logger = logger
        self.feature_extractors = ["image"]
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        if train:
            self.model = self.build()
        else:
            self.model = self.load_model(MODEL_PATH)
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.steps_per_epoch = STEPS_PER_EPOCH

    def build(self):
        """
        Build neural network model
        
        Returns 
        -------
        keras.models.Model : 
            neural network model
        """
        image_input_layer = Input(shape=self.input_shape)
        image_layer = Conv2D(32, (7, 7), activation='relu',padding= "valid",kernel_initializer="glorot_normal")(image_input_layer)
        image_layer = Dropout(.2)(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(64,(5,5),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(image_layer)
        image_layer = Dropout(.2)(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(128,(3,3),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(image_layer)
        image_layer = Dropout(.2)(image_layer)

        image_layer = Flatten()(image_layer)

        dlib_points_input_layer = Input(shape=(1,68,2))
        dlib_points_layer = Conv2D(32, (1, 3), activation='relu',padding= "valid",kernel_initializer="glorot_normal")(dlib_points_input_layer)
        dlib_points_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_layer)
        dlib_points_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(dlib_points_layer)
        dlib_points_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_layer)
        dlib_points_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(dlib_points_layer)

        dlib_points_layer = Flatten()(dlib_points_layer)

        dlib_points_dist_input_layer = Input(shape=(1,68,1))
        dlib_points_dist_layer = Conv2D(32, (1, 3), activation='relu',padding= "valid",kernel_initializer="glorot_normal")(dlib_points_dist_input_layer)
        dlib_points_dist_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_dist_layer)
        dlib_points_dist_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer='glorot_normal')(dlib_points_dist_layer)
        dlib_points_dist_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_dist_layer)
        dlib_points_dist_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer='glorot_normal')(dlib_points_dist_layer)

        dlib_points_dist_layer = Flatten()(dlib_points_dist_layer)

        dlib_points_angle_input_layer = Input(shape=(1,68,1))
        dlib_points_angle_layer = Conv2D(32, (1, 3), activation='relu',padding= "valid",kernel_initializer="glorot_normal")(dlib_points_angle_input_layer)
        dlib_points_angle_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_angle_layer)
        dlib_points_angle_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer='glorot_normal')(dlib_points_angle_layer)
        dlib_points_angle_layer = MaxPooling2D(pool_size=(1, 2))(dlib_points_angle_layer)
        dlib_points_angle_layer = Conv2D(64,(1, 3),activation = "relu",padding="valid",kernel_initializer='glorot_normal')(dlib_points_angle_layer)

        dlib_points_angle_layer = Flatten()(dlib_points_angle_layer)

        merged_layers = keras.layers.concatenate([image_layer, dlib_points_layer,dlib_points_dist_layer,dlib_points_angle_layer])
        
        merged_layers = Dense(252, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(1024, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(self.number_of_class, activation='softmax')(merged_layers)
        
        self.model = Model(inputs=[image_input_layer, dlib_points_input_layer,dlib_points_dist_input_layer,dlib_points_angle_input_layer],outputs=merged_layers)
        self.built = True
        return self.model

    

    def train(self):
        """Traines the neuralnet model.      
        This method requires the following two directory to exist
        /PATH-TO-DATASET-DIR/train
        /PATH-TO-DATASET-DIR/test
        
        """
        assert self.built == True , "Model not built yet."
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(LEARNING_RATE),
                    metrics=['accuracy'])
        # self.model.fit(x_train,y_train,epochs = EPOCHS, 
        #                 batch_size = BATCH_SIZE,validation_data=(x_test,y_test))
        self.preprocessor = self.preprocessor(DATA_SET_DIR)
        self.model.fit_generator(self.preprocessor.flow(),steps_per_epoch=self.steps_per_epoch,
                        epochs=self.epochs,
                        validation_data=([self.preprocessor.test_images,self.preprocessor.test_dpoints,self.preprocessor.dpointsDists,self.preprocessor.dpointsAngles], self.preprocessor.test_image_emotions))
        score = self.model.evaluate([self.preprocessor.test_images,self.preprocessor.test_dpoints,self.preprocessor.dpointsDists,self.preprocessor.dpointsAngles], self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.models_local_folder, score)

    def predict(self,face):
        assert face.shape == IMG_SIZE, "Face image size should be "+str(IMG_SIZE)
        face = face.reshape(-1,48,48,1)
        emotions = self.model.predict(face)[0]
        return emotions