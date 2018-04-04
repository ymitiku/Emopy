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
    def __init__(self,input_shape,convnet_model_path=None,preprocessor = None,logger=None,train=True,max_sequence_length=71):
        self.convnet_model_path = convnet_model_path;
        self.max_sequence_length = max_sequence_length

        NeuralNet.__init__(self,input_shape,preprocessor,logger,train)
        self.models_local_folder = "rnn"
        self.logs_local_folder = self.models_local_folder
        if not os.path.exists(os.path.join(LOG_DIR,self.logs_local_folder)):
            os.makedirs(os.path.join(LOG_DIR,self.logs_local_folder))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt")])
            print("Logging to file",os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt"))
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
    def predict(self,sequence_faces):
        assert sequence_faces[0].shape == IMG_SIZE, "Face image size should be "+str(IMG_SIZE)
        face = face.reshape(-1,self.max_sequence_length,48,48,1)
        emotions = self.model.predict(face)[0]
        return emotions
    def process_web_cam(self):
        model = model_from_json(open("models/rnn/rnn-0.json").read())
        model.load_weights("models/rnn/rnn-0.h5")
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        sequences = np.zeros((self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]) )
        while cap.isOpened():
            while len(sequences)<self.max_sequence_length:
                ret, frame = cap.read()
                frame = cv2.resize(frame,(300,240))
                faces,rectangles = self.preprocessor.get_faces(frame,face_detector)
                face = faces[0]
                sequences
            predictions = []
            for i in range(len(faces)):
                face = preprocessor.sanitize(faces[i])
                predictions.append(neuralNet.predict(face))

            self.postProcessor = self.postProcessor(img,rectangles,predictions)
            cv2.imshow("Image",img)
            if (cv2.waitKey(10) & 0xFF == ord('q')):
                break
        cv2.destroyAllWindows()
    def train(self):
        from sklearn.metrics import confusion_matrix
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
        
        # self.model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        print self.model.output.shape

        self.model.fit_generator(self.preprocessor.flow(),steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        validation_data=(self.preprocessor.test_sequences_dpoints, self.preprocessor.test_sequence_labels))
        
        score = self.model.evaluate(self.preprocessor.test_sequences, self.preprocessor.test_sequence_labels)
        self.save_model()
        self.logger.log_model(self.models_local_folder, score)
    def predict(self,sequence_faces):
        assert sequence_faces[0].shape == IMG_SIZE, "Face image size should be "+str(IMG_SIZE)
        face = face.reshape(-1,self.max_sequence_length,48,48,1)
        emotions = self.model.predict(face)[0]
        return emotions



class DlibLSTMNet(LSTMNet):
    def __init__(self,input_shape,convnet_model_path=None,preprocessor = None,logger=None,train=True):
        LSTMNet.__init__(self,input_shape,convnet_model_path,preprocessor,logger,train)
       
        self.models_local_folder = "drnn"
        self.logs_local_folder = self.models_local_folder
        if not os.path.exists(os.path.join(LOG_DIR,self.logs_local_folder)):
            os.makedirs(os.path.join(LOG_DIR,self.logs_local_folder))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt")])
            print("Logging to file",os.path.join(LOG_DIR,self.logs_local_folder,self.logs_local_folder+".txt"))
        else:
            self.logger = logger
        self.model = self.build()

        
    def build(self):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (3, 1), padding='valid', activation='relu'),input_shape=(self.max_sequence_length, 68, 2, 1)))
        model.add(TimeDistributed(Conv2D(64, (3, 1), padding='valid', activation='relu')))
        # model.add(TimeDistributed(Dropout(0.5)))
        # model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
        model.add(TimeDistributed(Flatten()))
        # model.add(Bidirectional(LSTM(16,return_sequences=True,stateful=False)))
        model.add(LSTM(32,return_sequences=False,stateful=False))
        model.add(Dense(6,activation="softmax"))
        
        return model;

    
    def predict(self,dlib_features):
        emotions = self.model.predict(dlib_features)
        return emotions
    def load_model(self,path_path):
        with open(path_path+".json") as json_file:
            model = model_from_json(json_file.read())
            model.load_weights(path_path+".h5")
            return model
    
