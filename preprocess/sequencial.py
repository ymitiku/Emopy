from preprocess.base import Preprocessor
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.utils  import shuffle
from train_config import BATCH_SIZE


class SequencialPreprocessor(Preprocessor):

    def __init__(self,classifier, input_shape = None,batch_size=BATCH_SIZE,augmentation = False,verbose = True,max_sequence_length=64):
        Preprocessor.__init__(self,classifier,input_shape,batch_size,augmentation,verbose)
        self.max_sequence_length = max_sequence_length

    def get_sequences_images(self,sequences_dirs):
        output = np.zeros((len(sequences_dirs),self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
        for i in range(len(sequences_dirs)):
            output[i] = self.get_sequence_images(sequences_dirs[i])
        return output
    def get_sequence_images(self,sequence_dir):
        files = os.listdir(sequence_dir)
        if len(files)==0:
            raise Exception("Sequence "+sequence_dir+" Contains no images")
        
        output = np.zeros((self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
        index = 0
        files.sort()
        steps =float(len(files))/self.max_sequence_length
        currentOutputIndex = 0
        while int(index)<len(files) and currentOutputIndex<self.max_sequence_length:
            img = cv2.imread(os.path.join(sequence_dir,files[int(index)]))
            img = self.sanitize(img).reshape(self.input_shape)
            output[currentOutputIndex] = img
            index +=steps
            currentOutputIndex +=1
        # pad with last image
        currentOutputIndex = int(currentOutputIndex)
        last_image = np.array(img,copy=True)
        while currentOutputIndex<self.max_sequence_length:
            output[currentOutputIndex] = last_image
            currentOutputIndex +=1
        return output
    def load_dataset(self,path):
        assert os.path.exists(path),"Specified dataset directory '"+path+"' does not exist "
        self.train_sequences = []
        self.train_sequence_labels = []
        for emdir in os.listdir(os.path.join(path,"train")):
            if emdir == "contempt":
                continue
            print("Loading ",os.path.join(path,"train",emdir))
            for sequence in os.listdir(os.path.join(path,"train",emdir)):
                self.train_sequences += [sequence]
                self.train_sequence_labels +=[emdir]
        self.train_sequences = np.array(self.train_sequences)
        self.train_sequence_labels = np.array(self.train_sequence_labels)
        test_seq = []
        test_seq_label = []
        for emdir in os.listdir(os.path.join(path,"test")):
            if emdir == "contempt":
                continue
            print("Loading ",os.path.join(path,"test",emdir))
            for sequence in os.listdir(os.path.join(path,"test",emdir)):
                test_seq += [sequence]
                test_seq_label +=[emdir]

        

        self.test_sequences = np.zeros((len(test_seq),self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
        num_class=self.classifier.get_num_class()
        self.test_sequence_labels = np.zeros((len(test_seq_label),6))

        for i in range(len(test_seq)):
            self.test_sequences[i] = self.get_sequence_images(os.path.join(path,"test",test_seq_label[i],test_seq[i]))
            self.test_sequence_labels[i] = np.eye(6)[self.classifier.get_class(test_seq_label[i])]
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
                    sequences_dirs.append(os.path.join(self.dataset_path,"train",sequences_labels[j],sequences[j]))
                X = self.get_sequences_images(sequences_dirs)
                y = np.zeros((len(sequences_labels),6))
                for k  in range(len(sequences_labels)):                
                    y[k] = np.eye(6)[self.classifier.get_class(sequences_labels[k])]
                X = X.astype(np.float32)/255;
                yield X,y

