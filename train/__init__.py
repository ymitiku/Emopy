import argparse

from nets.base import NeuralNet
from util import SevenEmotionsClassifier,PositiveNeutralClassifier,PositiveNegetiveClassifier
from preprocess.base import Preprocessor
from preprocess.dlib_input import DlibInputPreprocessor
from nets.dlib_inputs import DlibPointsInputNeuralNet
from preprocess.multinput import MultiInputPreprocessor
from nets.multinput import MultiInputNeuralNet
from nets.vggface import VGGFaceEmopyNet
from loggers.base import EmopyLogger
from nets.rnn import LSTMNet,DlibLSTMNet
from preprocess.sequencial import SequencialPreprocessor
from preprocess.sequencial import DlibSequencialPreprocessor
from train_config import DATA_SET_DIR,AUGMENTATION,BATCH_SIZE,EPOCHS,IMG_SIZE,INPUT_SHAPE,LEARNING_RATE
from train_config import NETWORK_TYPE,STEPS_PER_EPOCH
from config import EMOTION_CLASSIFICATION

def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n","--net",default="face",type=str)
    parser.add_argument("-c","--emotions",default=EMOTION_CLASSIFICATION,type=str)
    parser.add_argument("-b","--batch",default=BATCH_SIZE,type=int)
    parser.add_argument("-e","--epochs",default=EPOCHS,type=int)
    parser.add_argument("-s","--steps",default=STEPS_PER_EPOCH,type=int)
    parser.add_argument("-l","--lr",default=LEARNING_RATE,type=float)
    parser.add_argument("-d","--dataset_path",default=DATA_SET_DIR,type=str)
    parser.add_argument("-m","--model_path",default="models/model",type=str)
    parser.add_argument("-a","--augmentation",default=AUGMENTATION,type=bool)
    parser.add_argument("-v","--verbose",default=False,type=bool)
    parser.add_argument("-x","--sequence_length",default=71,type=int)

    args = parser.parse_args()
    return args

def get_network(args):
    
    if args.emotions=="all":
        classifier = SevenEmotionsClassifier()
    elif args.emotions=="pos-neg":
        classifier = PositiveNeutralClassifier()
    elif args.emotions=="pos-neu":
        classifier = PositiveNegetiveClassifier()
    else:
        raise Exception("emotions should be one of all,pos-neg or pos-neu. But it is "+str(args.emotions))
    input_shape = (48,48,1)

    if args.net=="face":
        preprocessor = Preprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose)
        lgr = EmopyLogger(["logs/log.txt"])
        net = NeuralNet((48,48,1),preprocessor,logger=lgr,train=True)
        return net
    elif args.net=="dlib":
        preprocessor = DlibInputPreprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose)
        lgr = EmopyLogger(["logs/log.txt"])
        net = DlibPointsInputNeuralNet((48,48,1),preprocessor,logger=lgr,train=True)
        return net
    elif args.net == "face+dlib":
        preprocessor = MultiInputPreprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose)
        lgr = EmopyLogger(["logs/log.txt"])
        net = MultiInputNeuralNet((48,48,1),preprocessor,logger=lgr,train=True)
        return net
    elif args.net == "vgg-face":
        input_shape = (48,48,3)
        preprocessor = Preprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose)
        lgr = EmopyLogger(["logs/log.txt"])
        net = VGGFaceEmopyNet((48,48,1),preprocessor,logger=lgr,train=True)
        return net
    elif args.net == "rnn":
        preprocessor = SequencialPreprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose,max_sequence_length=args.sequence_length)
        lgr = EmopyLogger(["logs/log.txt"])
        net = LSTMNet(input_shape,preprocessor=preprocessor,logger=lgr,train=True,max_sequence_length=args.sequence_length)
        return net
    elif args.net == "dlib-rnn":
        preprocessor = DlibSequencialPreprocessor(classifier,input_shape=input_shape,batch_size=args.batch,augmentation=args.augmentation,verbose=args.verbose,max_sequence_length=args.sequence_length)
        lgr = EmopyLogger(["logs/log.txt"])
        net = DlibLSTMNet(input_shape,preprocessor=preprocessor,logger=lgr,train=True,max_sequence_length=args.sequence_length)
        return net
    else:
        raise Exception("net arg should be one of face,face+dlib,vgg-face,rnn or dlib-rnn, but it is "+str(args.net))
