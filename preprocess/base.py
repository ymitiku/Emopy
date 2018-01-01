from __future__ import print_function
import os
from sklearn.utils import shuffle
import numpy as np
import cv2
import dlib
from keras.preprocessing.image import ImageDataGenerator
from feature_extraction import ImageFeatureExtractor

class Preprocessor(object):
    """"Base class for preprocessors.
    Preprocessing includes reading and sanitizing dataset.
    
    parameters
    ---------
    input_shape : tuple
        shape of input images to generate
    classifier : util.Classifier
        classifier used for classifying emotions
    batch_size : int
        batch size for generating batch of images
    """
    
    def __init__(self,classifier, input_shape = None,batch_size=32,augmentation = False,verbose = True):
        self.classifier = classifier
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.called = False
        self.verbose = verbose
        self.augmentation = augmentation
        if augmentation:
            self.datagenerator = ImageDataGenerator(
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                # shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip=True,
                data_format="channels_last"
                 
            )
        self.feature_extractor = ImageFeatureExtractor()
    """Load dataset with given path
    
    parameters
    ----------
    path    : str
        path to directory containing training and test directory.
    """
    def load_dataset(self,path):
        assert os.path.exists(path),"Specified dataset directory '"+path+"' does not exist "
        train_test_dir = os.listdir(path)
        assert "train" in train_test_dir , "Specified dataset directory '"+path+"' does not  contain train directory." 
        assert "test" in train_test_dir , "Specified dataset directory '"+path+"' does not  contain test directory." 
        self.train_image_paths = []
        self.train_image_emotions = []
        for emdir in os.listdir(os.path.join(path,"train")):
            print("Loading ",os.path.join(path,"train",emdir))
            for img_file in os.listdir(os.path.join(path,"train",emdir)):
                self.train_image_paths.append(os.path.join(path,"train",emdir,img_file))
                self.train_image_emotions.append(self.classifier.get_class(emdir))
        self.test_image_paths = []
        self.test_image_emotions = []
        for emdir in os.listdir(os.path.join(path,"test")):
            print("Loading ",os.path.join(path,"test",emdir))
            for img_file in os.listdir(os.path.join(path,"test",emdir)):
                self.test_image_paths.append(os.path.join(path,"test",emdir,img_file))
                self.test_image_emotions.append(self.classifier.get_class(emdir))
        assert len(self.train_image_emotions) == len(self.train_image_paths), "number of train inputs are not equal to train labels"
        assert len(self.test_image_emotions) == len(self.test_image_paths), "number of test inputs are not equal to test labels"
        self.train_image_emotions = np.array(self.train_image_emotions)
        self.train_image_paths = np.array(self.train_image_paths)
        self.test_images = self.get_images(self.test_image_paths).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2])
        self.test_image_emotions = np.eye(self.classifier.get_num_class()) [np.array(self.test_image_emotions)]
    """Preprocess given path
    
    parameters
    ----------
    path    : str
        path to directory containing training and test directory.
    """
    def __call__(self,path):
        self.load_dataset(path)        
        self.test_images = self.feature_extractor.extract(self.test_images);
        self.called = True
        return self
    def generate_indexes(self,random=True):
        indexes = range(len(self.train_image_emotions))
        if (random):
            indexes = shuffle(indexes)
        indexes = np.array(indexes)
        return indexes
    def flow(self):
        assert self.called, "Preprocessor should be called with path of dataset first to use flow method."
        while True:
            indexes = self.generate_indexes(True)
            for i in range(0,len(indexes) - self.batch_size,self.batch_size):
                current_indexes = indexes[i:i+self.batch_size]
                current_paths = self.train_image_paths[current_indexes]
                current_emotions = self.train_image_emotions[current_indexes]
                current_images = self.get_images(current_paths,self.augmentation).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2])
                current_images = self.feature_extractor.extract(current_images)
                current_emotions = np.eye(self.classifier.get_num_class())[current_emotions]
                yield current_images,current_emotions
    
    def sanitize(self,image):
        if image is None:
            raise "Unable to sanitize None image"
        if len(image.shape)>2 and self.input_shape[2]==1:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dim = (self.input_shape[0],self.input_shape[1])
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA).astype(np.float32)
        return image
    
    
    def get_images(self,paths,augmentation = False):
        output = np.zeros(shape=(len(paths),self.input_shape[0],self.input_shape[1],self.input_shape[2]),dtype=np.uint8)
        for i in range(len(paths)):
            img = cv2.imread(paths[i],cv2.IMREAD_COLOR)
            img = self.sanitize(img)
            img = img.reshape(self.input_shape)
            if(augmentation):
                # print ("augmentation")
                if (self.input_shape[2]==3):
                    img_shape = img.shape
                    r = img[:,:,0]
                    g = img[:,:,1]
                    b = img[:,:,2]
                    r = r.reshape(-1,img_shape[0],img_shape[1])
                    g = g.reshape(-1,img_shape[0],img_shape[1])
                    b = b.reshape(-1,img_shape[0],img_shape[1])

                    r = self.datagenerator.random_transform(r)
                    g = self.datagenerator.random_transform(g)
                    b = self.datagenerator.random_transform(b)

                    r = r.reshape(-1,img_shape[0],img_shape[1],1)
                    g = g.reshape(-1,img_shape[0],img_shape[1],1)
                    b = b.reshape(-1,img_shape[0],img_shape[1],1)
                    img = np.concatenate((r, g, b),axis=-1)
                elif self.input_shape[2]==1:
                    img_shape = img.shape
                    # print (img.shape,self.input_shape)
                    # img = img.reshape(self.input_shape[0],self.input_shape[1])
                    # print("Imageshape",img.shape)
                    img = self.datagenerator.random_transform(img)
                    img = img.reshape(img_shape)
                else:
                    raise Exception("Image should be either gray or rgb. Image shape is:",img.shape)
                
            output[i] = img
        return output
    
    def get_faces(self,frame,detector):
        faces = detector(frame)
        output = []
        rectangles =[]
        for face in faces:
            top = max(0,face.top())
            left = max(0,face.left())
            bottom = min(frame.shape[0],face.bottom())
            right = min(frame.shape[1],face.right())
            rectangles.append(dlib.rectangle(left,top,right,bottom));
            output.append(frame[top:bottom, left:right])
        return output,rectangles
    def load_sequencial_dataset(self,path,max_sequence_length=71):
        path = "dataset/ck-sequence"
        X = []
        Y = []
        for em_dir in os.listdir(path):
            if  em_dir == "contempt":
                continue
            for sequence in os.listdir(os.path.join(path,em_dir)):
                x = np.zeros((max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
                currentIndex = 0
                # y = []
                for img_file in os.listdir(os.path.join(path,em_dir,sequence)):
                    img = cv2.imread(os.path.join(path,em_dir,sequence,img_file))
                    img = self.sanitize(img).reshape(self.input_shape[0],self.input_shape[1],self.input_shape[2])
                    # x[currentIndex] = img.reshape(img.shape[0]*img.shape[1])
                    x[currentIndex] = img
                    # y += [np.eye(7)[self.classifier.get_class(em_dir)]]
                    currentIndex += 1
                    if currentIndex ==  max_sequence_length:
                        break
                if currentIndex>max_sequence_length:
                    raise Exception("Sequence with : "+str(currentIndex)+" length found")
                last_image = x[currentIndex-1]
                for i in range(currentIndex,max_sequence_length):
                    x[i] = last_image
                    # y += [np.eye(7)[self.classifier.get_class(em_dir)]]
                X+=[x]
                # Y+=[y]
                
                Y+=[np.eye(6)[self.classifier.get_class(em_dir)]]
            print ("loaded",em_dir ) 
        print ("sequences",len(X))
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
            