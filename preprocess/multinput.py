from __future__ import print_function
import os
from sklearn.utils import shuffle
import numpy as np
import cv2
import dlib
from keras.preprocessing.image import ImageDataGenerator
from preprocess.base import Preprocessor
from preprocess.feature_extraction import DlibFeatureExtractor
from config import SHAPE_PREDICTOR_PATH

class MultiInputPreprocessor(Preprocessor):
    """"Preprocessor extracts dlib points, dlib points distance and dlib points angle from 
    centroid.
    
    parameters
    ---------
    input_shape : tuple
        shape of input images to generate
    classifier : util.Classifier
        classifier used for classifying emotions
    batch_size : int
        batch size for generating batch of images
    verbose    : boolean
        if true print logs to screen
    """
    
    def __init__(self,classifier, input_shape = None,batch_size=32,verbose = True):
        Preprocessor.__init__(self,classifier,input_shape,batch_size,verbose)
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.feature_extractor = DlibFeatureExtractor(self.predictor)
    def __call__(self,path):
        super(MultiInputPreprocessor,self).__call__(path);
        self.test_images,self.test_dpoints,self.dpointsDists,self.dpointsAngles = self.feature_extractor.extract(self.test_images);

    
    def flow(self):
        assert self.called, "Preprocessor should be called with path of dataset first to use flow method."
        while True:
            indexes = self.generate_indexes(True)
            for i in range(0,len(indexes) - self.batch_size,self.batch_size):
                current_indexes = indexes[i:i+self.batch_size]
                current_paths = self.train_image_paths[current_indexes]
                current_emotions = self.train_image_emotions[current_indexes]
                current_images = self.get_images(current_paths).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2])
                current_images,dpoints,dpointsDists,dpointsAngles = self.feature_extractor.extract(current_images)
                current_emotions = np.eye(self.classifier.get_num_class())[current_emotions]
                yield [current_images,dpoints,dpointsDists,dpointsAngles],current_emotions
   
    
    
    
    
    