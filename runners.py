from config import SESSION
from nets.base import NeuralNet
from constants import IMG_SIZE
from test_config import TEST_TYPE, TEST_IMAGE
from preprocess.dataset_process import sanitize, get_faces
from postprocess.base import PostProcessor
import cv2
import dlib


def run():
    if SESSION == 'train':
        run_train()
    elif SESSION == 'test':
        run_test()
def run_train():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    neuralNet = NeuralNet(input_shape)
    neuralNet.train()

def run_test():
    input_shape = (IMG_SIZE[0],IMG_SIZE[1],1)
    neuralNet = NeuralNet(input_shape,train = False)
    postProcessor = PostProcessor()
    face_detector = dlib.get_frontal_face_detector()

    if TEST_TYPE=="image":
        img = cv2.imread(TEST_IMAGE)
        faces = get_faces(img,face_detector)
        for face in faces:
            f = sanitize(face)
            predictions = neuralNet.predict(f)
            emotion = postProcessor.arg_max(predictions[0])
            emotion_string = postProcessor.emotion2string(emotion)
            print emotion_string
            print predictions
        # print predictions
    elif TEST_TYPE =="video":
        pass
    elif TEST_TYPE == "webcam":
        pass


    