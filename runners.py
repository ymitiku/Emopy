from config import SESSION,IMG_SIZE
from nets.base import NeuralNet
from test_config import TEST_TYPE, TEST_IMAGE
import cv2
import dlib
from util import SevenEmotionsClassifier
from preprocess.base import Preprocessor
from postprocess.base import PostProcessor
from preprocess.multinput import MultiInputPreprocessor
from nets.multinput import MultiInputNeuralNet
from nets.rnn import LSTMNet,DlibLSTMNet
from train_config import NETWORK_TYPE,AUGMENTATION
from preprocess.sequencial import SequencialPreprocessor,DlibSequencialPreprocessor
from multiprocessing.queues import Queue
from threading import Thread
import numpy as np
from preprocess.dlib_input import DlibInputPreprocessor
from nets.dlib_inputs import DlibPointsInputNeuralNet
maxSequenceLength = 10
from test_config import MODEL_PATH
from train_config import DATA_SET_DIR,EPOCHS,LEARNING_RATE,STEPS_PER_EPOCH,AUGMENTATION,BATCH_SIZE

def run():
    if SESSION == 'train':
        run_train()
    elif SESSION == 'test':
        run_test()
def run_train():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    classifier = SevenEmotionsClassifier()
    if(NETWORK_TYPE == "mi"):
        preprocessor = MultiInputPreprocessor(classifier,input_shape = input_shape,augmentation = AUGMENTATION)
        neuralNet = MultiInputNeuralNet(input_shape,preprocessor=preprocessor,train=True)
    elif NETWORK_TYPE == "si":
        preprocessor = Preprocessor(classifier,input_shape = input_shape,augmentation = AUGMENTATION)
        neuralNet = NeuralNet(input_shape,preprocessor=preprocessor,train=True)
    elif NETWORK_TYPE =="rnn":
        preprocessor = SequencialPreprocessor(classifier,input_shape = input_shape,augmentation = AUGMENTATION)("dataset/ck-split")
        neuralNet = LSTMNet(input_shape,preprocessor=preprocessor,train=True)
    elif NETWORK_TYPE =="drnn":
        preprocessor = DlibSequencialPreprocessor(classifier,input_shape = input_shape,augmentation = AUGMENTATION)("dataset/ck-split")
        neuralNet = DlibLSTMNet(input_shape,preprocessor=preprocessor,train=True)
    elif NETWORK_TYPE =="dinn":
        preprocessor = DlibInputPreprocessor(classifier,input_shape = input_shape,augmentation = AUGMENTATION)
        neuralNet = DlibPointsInputNeuralNet(input_shape,preprocessor=preprocessor,train=True)

    neuralNet.train()


def arg_max(array):
        max_value = array[0]
        index = 0
        for i,el in enumerate(array):
            if el > max_value:
                index = i
                max_value = el
        return index



def draw_landmarks(frame,landmarks):
    for i in range(len(landmarks)):
        landmark = landmarks[i]
        cv2.circle(frame,(int(landmark[0]),int(landmark[1])),1,color=(255,0,0),thickness=1)

def run_test():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)   
    classifier = SevenEmotionsClassifier()
    preprocessor = Preprocessor(classifier,input_shape = input_shape)
    postProcessor = PostProcessor(classifier)
    neuralNet = NeuralNet(input_shape,preprocessor = preprocessor,train = False)
    face_detector = dlib.get_frontal_face_detector()

    if TEST_TYPE=="image":
        img = cv2.imread(TEST_IMAGE)
        faces,rectangles = preprocessor.get_faces(img,face_detector)
        predictions = []
        for i in range(len(faces)):
            face = preprocessor.sanitize(faces[i])
            predictions.append(neuralNet.predict(face))

        postProcessor = postProcessor(img,rectangles,predictions)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif TEST_TYPE =="video":
        pass
    elif TEST_TYPE == "webcam":
        if NETWORK_TYPE == "drnn":
            # cap = cv2.VideoCapture("/home/mtk/iCog/projects/emopy/test-videos/75Emotions.mp4")
            cap = cv2.VideoCapture(-1)
            preprocessor = DlibSequencialPreprocessor(classifier,input_shape = input_shape)
            neuralNet = DlibLSTMNet(input_shape,preprocessor = preprocessor,train = False)
            face_detector = dlib.get_frontal_face_detector()
            postProcessor = PostProcessor(classifier)
            # if cap.isOpened():
            #     recognitionThread = Thread(target=start_recognition_task,args=(preprocessor,neuralNet))
            #     recognitionThread.start()
            print "opening camera"

            
            current_sequence = np.zeros((maxSequenceLength,68,2,1))
            currentIndex = 0


            
            currentEmotion = ""
            while cap.isOpened():
                ret,frame = cap.read()
                currentWidth = frame.shape[1]
                width = 600
                ratio = currentWidth/float(width)
                height = frame.shape[0]/float(ratio)
                frame = cv2.resize(frame,(width,int(height)))
                faces,rectangles = preprocessor.get_faces(frame,face_detector)
                if(len(faces)>0):
                    face,rectangle = faces[0],rectangles[0]
                    face = preprocessor.sanitize(face)
                    dlib_points = preprocessor.get_face_dlib_points(face)
                    # draw_landmarks(face,dlib_points)
                    # cv2.imshow("Face",face)
                    current_sequence[currentIndex:currentIndex+2] = [np.array(np.expand_dims(dlib_points,2)),np.expand_dims(dlib_points,2)]
                    currentIndex+=2
                    # sequencialQueue.put(face)
                    postProcessor.overlay(frame, [rectangle], [currentEmotion])
                else:
                    current_sequence = np.zeros((maxSequenceLength,68,2,1))
                    currentIndex = 0
                if currentIndex > maxSequenceLength-2:
                    current_sequence= current_sequence.astype(np.float32)/IMG_SIZE[0]
                    predictions = neuralNet.predict(np.expand_dims(current_sequence,0))[0]
                    print predictions
                    emotion = arg_max(predictions)
                    currentEmotion = preprocessor.classifier.get_string(emotion)
                    current_sequence = np.zeros((maxSequenceLength,68,2,1))
                    currentIndex = 0
                cv2.imshow("Webcam",frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break
            cv2.destroyAllWindows()
        else:
            input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)   
            classifier = SevenEmotionsClassifier()
            preprocessor = Preprocessor(classifier,input_shape = input_shape)
            postProcessor = PostProcessor(classifier)
            neuralNet = NeuralNet(input_shape,preprocessor = preprocessor,train = False)
            face_detector = dlib.get_frontal_face_detector()
            neuralNet.load_model(MODEL_PATH)
            # cap = cv2.VideoCapture(-1)
            cap = cv2.VideoCapture("/home/mtk/iCog/projects/emopy/test-videos/75Emotions.mp4")
            while cap.isOpened():
                ret,frame = cap.read()
                currentWidth = frame.shape[1]
                width = 600
                ratio = currentWidth/float(width)
                height = frame.shape[0]/float(ratio)
                frame = cv2.resize(frame,(width,int(height)))
                faces,rectangles = preprocessor.get_faces(frame,face_detector)
                if(len(faces)>0):
                    emotions = []
                    for i in range(len(faces)):
                        print faces[i].shape
                        face = preprocessor.sanitize(faces[i]).astype(np.float32)/255;
                        print face.shape
                        predictions = neuralNet.predict(face.reshape(-1,48,48,1))[0]
                        print predictions
                        emotions.append(classifier.get_string(arg_max(predictions)))

                    postProcessor.overlay(frame,rectangles,emotions)
                cv2.imshow("Webcam",frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break
            cv2.destroyAllWindows()

def start_train_program(network_type=NETWORK_TYPE ,dataset_dir=DATA_SET_DIR,epochs=EPOCHS,batch_size=BATCH_SIZE,lr=LEARNING_RATE,steps=STEPS_PER_EPOCH,augmentation=AUGMENTATION):
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    classifier = SevenEmotionsClassifier()
    if(network_type == "mi"):
        preprocessor = MultiInputPreprocessor(classifier,input_shape = input_shape,batch_size=batch_size,augmentation = augmentation)
        neuralNet = MultiInputNeuralNet(input_shape,preprocessor=preprocessor,
        learning_rate= lr,batch_size = batch_size,epochs = epochs,steps_per_epoch = steps,dataset_dir= dataset_dir
        )
    elif network_type == "si":
        preprocessor = Preprocessor(classifier,input_shape = input_shape,batch_size=batch_size,augmentation = augmentation)
        neuralNet = NeuralNet(input_shape,preprocessor=preprocessor,train=True)
    elif network_type =="rnn":
        preprocessor = SequencialPreprocessor(classifier,input_shape = input_shape,batch_size=batch_size,augmentation = augmentation)("dataset/ck-split")
        neuralNet = LSTMNet(input_shape,preprocessor=preprocessor,train=True)
    elif network_type =="drnn":
        preprocessor = DlibSequencialPreprocessor(classifier,input_shape = input_shape,batch_size=batch_size,augmentation = augmentation)("dataset/ck-split")
        neuralNet = DlibLSTMNet(input_shape,preprocessor=preprocessor,train=True)
    elif network_type =="dinn":
        preprocessor = DlibInputPreprocessor(classifier,input_shape = input_shape,batch_size=batch_size,augmentation = augmentation)
        neuralNet = DlibPointsInputNeuralNet(input_shape,preprocessor=preprocessor,train=True)

    neuralNet.train()
def start_test_program():
    pass



