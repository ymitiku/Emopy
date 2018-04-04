from preprocess.base import Preprocessor
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.utils  import shuffle
from train_config import BATCH_SIZE

from keras.preprocessing.image import ImageDataGenerator
import dlib
from preprocess.feature_extraction import DlibFeatureExtractor
from config import IMG_SIZE

class SequencialPreprocessor(Preprocessor):

    def __init__(self,classifier, input_shape = None,batch_size=BATCH_SIZE,augmentation = False,verbose = True,max_sequence_length=6):
        Preprocessor.__init__(self,classifier,input_shape,batch_size,augmentation,verbose)
        self.max_sequence_length = max_sequence_length
        self.datagenerator = ImageDataGenerator(
                rotation_range = 40,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                shear_range = 0.1,
                zoom_range = 0.1,
                horizontal_flip=True,
                data_format="channels_last"
                 
            )
    def get_sequences_images(self,sequences_dirs,augmentation=False):
        output = np.zeros((len(sequences_dirs),self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))

        for i in range(len(sequences_dirs)):
            output[i] = self.get_sequence_images(sequences_dirs[i],augmentation)
        return output
    def get_sequence_images(self,sequence_dir,augmentation=False):
        files = os.listdir(sequence_dir)
        if len(files)==0:
            raise Exception("Sequence "+sequence_dir+" Contains no images")
        
        output = np.zeros((self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]),dtype=np.uint8)
        index = 0
        files.sort()
        steps = len(files)/float(self.max_sequence_length)
        currentStep = 0
        while index<len(files) and currentStep < self.max_sequence_length:
            img = cv2.imread(os.path.join(sequence_dir,files[int(index)]))
            img = self.sanitize(img).reshape(self.input_shape)
            if augmentation:
                img = self.datagenerator.random_transform(img)
            output[currentStep] = img
            index += steps
            currentStep +=1
        last_image = np.array(output[currentStep-1])
        while currentStep< self.max_sequence_length:
            output[currentStep] = np.array(last_image)
            currentStep+=1
        
        return output
    def load_dataset(self,path):
        assert os.path.exists(path),"Specified dataset directory '"+path+"' does not exist "
        all_sequences = []
        all_labels = []
        
        
        train_sequences = []
        train_sequence_labels = []

        for emdir in os.listdir(path):
            if emdir == "contempt":
                continue
            print("Loading ",os.path.join(path,emdir))
            for sequence in os.listdir(os.path.join(path,emdir)):

                all_sequences += [sequence]
                all_labels +=[emdir]


        (train_sequences,test_sequences,train_sequence_labels,test_sequence_labels) = train_test_split(all_sequences,all_labels) 
  
        self.train_sequences = np.array(train_sequences)
        self.train_sequence_labels = np.array(train_sequence_labels)

        self.test_sequences = np.zeros((len(test_sequences),self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
     
        self.test_sequence_labels = np.zeros((len(test_sequence_labels),6))

        for i in range(len(test_sequences)):
            self.test_sequences[i] = self.get_sequence_images(os.path.join(path,test_sequence_labels[i],test_sequences[i]))
            self.test_sequence_labels[i] = np.eye(6)[self.classifier.get_class(test_sequence_labels[i])]

        self.test_sequences = self.test_sequences.astype(np.float32)/255;
        self.test_sequences,self.test_sequence_labels = shuffle(self.test_sequences,self.test_sequence_labels)
        
    def __call__(self,path):
        self.load_dataset(path)
        self.called = True
        self.dataset_path = path
        return self
    def generate_indexes(self,random=True):
        indexes = range(len(self.train_sequences))
        if (random):
            indexes = shuffle(indexes)
        indexes = np.array(indexes)
        return indexes

    def flow(self):
        assert self.called, "Preprocessor should be called with path of dataset first to use flow method."
        while True:
            indexes = self.generate_indexes(True)
            for i in range(0,len(indexes) - self.batch_size,self.batch_size):
                currentIndexes = indexes[i:i+self.batch_size].tolist()
                sequences = self.train_sequences[currentIndexes]
                sequences_labels = self.train_sequence_labels[currentIndexes]
                sequences_dirs = []
                y = np.zeros((len(sequences_labels),6))
                for j in range(len(sequences)):
                    sequences_dirs.append(os.path.join(self.dataset_path,sequences_labels[j],sequences[j]))

                X = self.get_sequences_images(sequences_dirs,True)

                y = np.zeros((len(sequences_labels),6))
                for k  in range(len(sequences_labels)):                
                    y[k] = np.eye(6)[self.classifier.get_class(sequences_labels[k])]
                X = X.astype(np.float32)/255;
                yield X,y


class DlibSequencialPreprocessor(SequencialPreprocessor):
    def __init__(self,classifier, input_shape = None,batch_size=BATCH_SIZE,augmentation = False,verbose = True,max_sequence_length=71,predictor_path = "shape_predictor_68_face_landmarks.dat"):
        SequencialPreprocessor.__init__(self,classifier,input_shape,batch_size,augmentation,verbose,max_sequence_length)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
    

    def get_face_dlib_points(self,image):
        face = dlib.rectangle(0,0,image.shape[1]-1,image.shape[0]-1)
        img = image.reshape(IMG_SIZE[0],IMG_SIZE[1])
        shapes = self.predictor(img,face)
        parts = shapes.parts()
        output = np.zeros((68,2))
        for i,point in enumerate(parts):
            output[i]=[point.x,point.y]
        output = np.array(output).reshape((1,68,2,1))
        return output
    def get_dlib_points(self,images):
        output = np.zeros((len(images),68,2,1))
        for i in range(len(images)):
            output[i] = self.get_face_dlib_points(images[i])
        return output


    def flow(self):
        assert self.called, "Preprocessor should be called with path of dataset first to use flow method."
        while True:
            indexes = self.generate_indexes(True)
            for i in range(0,len(indexes) - self.batch_size,self.batch_size):
                currentIndexes = indexes[i:i+self.batch_size].tolist()
                sequences = self.train_sequences[currentIndexes]
                sequences_labels = self.train_sequence_labels[currentIndexes]
                sequences_dirs = []

               
                for j in range(len(sequences)):
                    sequences_dirs.append(os.path.join(self.dataset_path,sequences_labels[j],sequences[j]))
                X = self.get_sequences_images(sequences_dirs,True)
                y = np.zeros((len(sequences_labels),6))
                dlib_points = np.zeros((len(X),self.max_sequence_length,68,2,1))
                X = X.astype(np.uint8)
                for k  in range(len(sequences_labels)):                
                    y[k] = np.eye(6)[self.classifier.get_class(sequences_labels[k])]
                    dlib_points[k] = self.get_dlib_points(X[k])
                    

                dlib_points = dlib_points.astype(np.float32)
                dlib_points /= IMG_SIZE[0]
        
                yield dlib_points,y
            
    def load_dataset(self,path):
        assert os.path.exists(path),"Specified dataset directory '"+path+"' does not exist "
        all_sequences = []
        all_labels = []
        
        
        train_sequences = []
        train_sequence_labels = []

        for emdir in os.listdir(path):
            if emdir == "contempt":
                continue
            print("Loading ",os.path.join(path,emdir))
            for sequence in os.listdir(os.path.join(path,emdir)):

                all_sequences += [sequence]
                all_labels +=[emdir]


        (train_sequences,test_sequences,train_sequence_labels,test_sequence_labels) = train_test_split(all_sequences,all_labels) 
  
        self.train_sequences = np.array(train_sequences)
        self.train_sequence_labels = np.array(train_sequence_labels)

        test_sequences_array = np.zeros((len(test_sequences),self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
     
        self.test_sequence_labels = np.zeros((len(test_sequence_labels),6))
        print "loading test images"
        for i in range(len(test_sequences)):
            test_sequences_array[i] = self.get_sequence_images(os.path.join(path,test_sequence_labels[i],test_sequences[i]))
            self.test_sequence_labels[i] = np.eye(6)[self.classifier.get_class(test_sequence_labels[i])]
        output_test_sequences = np.zeros((len(test_sequences_array),self.max_sequence_length,68,2,1))
        for index in range(len(test_sequences_array)):
            output_test_sequences[index] = self.get_dlib_points(test_sequences_array[i].astype(np.uint8))
        
        self.test_sequences = output_test_sequences
        self.test_sequences /= IMG_SIZE[0]
        print "loaded test images"
        
            

