from preprocess.base import Preprocessor
from util import SevenEmotionsClassifier
from train_config import BATCH_SIZE,EPOCHS,STEPS_PER_EPOCH,LEARNING_RATE
from nets.base import NeuralNet

class EmopyProcess(object):
    def __init__(self,input_shape):
        self.classifier = SevenEmotionsClassifier()
        self.input_shape = input_shape
        self.preprocessor = Preprocessor(self.classifier,input_shape=self.input_shape,batch_size=BATCH_SIZE,augmentation=False,verbose=True)
        self.neuralNet = NeuralNet(self.input_shape,self.preprocessor)
    def start(self):
        pass
