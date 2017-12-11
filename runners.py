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


def run():
    if SESSION == 'train':
        run_train()
    elif SESSION == 'test':
        run_test()
def run_train():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    classifier = SevenEmotionsClassifier()
    preprocessor = MultiInputPreprocessor(classifier,input_shape = input_shape,augmentation = True)
    neuralNet = MultiInputNeuralNet(input_shape,preprocessor=preprocessor,train=True)
    neuralNet.train()


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
        pass


    