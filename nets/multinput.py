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

class CustomModelCheckPoint(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.last_loss = 1000000000
        self.last_accuracy = 0
        self.current_model_number = 0;
        self.epoch_number = 0
    # def on_train_begin(self,epoch, logs={}):
    #     return
 
    # def on_train_end(self, logs={}):
    #     return
 
    def on_epoch_begin(self,epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_number+=1
        current_val_loss = logs.get("val_loss")
        current_loss = logs.get("loss")
       

        if (self.last_loss-current_loss) > 0.01:
            current_weights_name = "weights"+str(self.current_model_number)+".h5"
            print("loss improved from "+str(self.last_loss)+" to "+str(current_loss)+", Saving model to "+current_weights_name)
            self.model.save_weights("models/"+current_weights_name);
            self.model.save_weights("models/last_weight.h5")
            self.current_model_number+=1
            self.last_loss = current_loss
            with open("log.txt","a+") as logfile:
                logfile.write("________________________________________________________\n")
                logfile.write("EPOCH    =")
                logfile.write(str(epoch)+"\n")
                logfile.write("TRAIN_LOSS =")
                logfile.write(str(current_loss)+"\n")
                logfile.write("VAL_LOSS =")
                logfile.write(str(current_val_loss)+"\n")
                logfile.write("---------------------------------------------------------\n")
                logfile.write("TRAIN_Age_LOSS  =")
                logfile.write(str(logs.get("age_estimation_loss"))+"\n")
                logfile.write("TRAIN_GENDER_LOSS =")
                logfile.write(str(logs.get("gender_probablity_loss"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("TRAIN_Age_ACC  =")
                logfile.write(str(logs.get("age_estimation_acc"))+"\n")
                logfile.write("TRAIN_GENDER_ACC =")
                logfile.write(str(logs.get("gender_probablity_acc"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("VAL_Age_LOSS  =")
                logfile.write(str(logs.get("val_age_estimation_loss"))+"\n")
                logfile.write("VAL_GENDER_LOSS =")
                logfile.write(str(logs.get("val_gender_probablity_loss"))+"\n")
                logfile.write("---------------------------------------------------------\n")

                logfile.write("VAL_Age_ACC  =")
                logfile.write(str(logs.get("val_age_estimation_acc"))+"\n")
                logfile.write("VAL_GENDER_ACC =")
                logfile.write(str(logs.get("val_gender_probablity_acc"))+"\n")

                logfile.write("********************************************************\n")
            with open("epoch_number.json","w+") as json_file:
                data = {"epoch_number":self.epoch_number}
                json.dump(data,json_file,indent=4)


class MultiInputNeuralNet(NeuralNet):
    """
    Neutral network whose inputs are images, dlib points, dlib points distances from centroid point
    and dlib points vector angle with respect to centroid vector.

    Parameters
    ----------
    input_shape : tuple
    
    """
    
    def __init__(self,input_shape,learning_rate,batch_size,epochs,steps_per_epoch,dataset_dir,preprocessor = None,logger=None,train=True):
        self.input_shape = input_shape
        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (48,48,1)" 
        self.models_local_folder = "minn"
        self.logs_local_folder = self.models_local_folder
        self.preprocessor = preprocessor
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.dataset_dir= dataset_dir
        
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

    def build(self):
        """
        Build neural network model
        
        Returns 
        -------
        keras.models.Model : 
            neural network model
        """
        image_input_layer = Input(shape=self.input_shape)
        image_layer = Conv2D(32, (3, 3), activation='relu',padding= "valid",kernel_initializer="glorot_normal")(image_input_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Dropout(0.2)(image_layer)
        image_layer = Conv2D(64,(3,3),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Conv2D(128,(3,3),activation = "relu",padding="valid",kernel_initializer="glorot_normal")(image_layer)
        image_layer = Dropout(0.2)(image_layer)
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
                    optimizer=keras.optimizers.Adam(self.learning_rate),
                    metrics=['accuracy'])
        # self.model.fit(x_train,y_train,epochs = EPOCHS, 
        #                 batch_size = BATCH_SIZE,validation_data=(x_test,y_test))
        self.preprocessor = self.preprocessor(self.dataset_dir)
        print "lr",self.learning_rate
        print "batch_size",self.batch_size
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