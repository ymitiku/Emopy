from process.base import EmopyProcess
from config import SESSION
from test_config import TEST_TYPE
from keras.models import model_from_json
from Queue import Queue
import cv2
import dlib
import numpy as np

class EmopySequencialProcess(EmopyProcess):
    def __init__(self,input_shape,max_sequence_length):
        self.max_sequence_length = max_sequence_length
        EmopyProcess.__init__(self,input_shape)
        self.model = self.load_model("models/rnn/rnn-10")
    def start(self):
        if SESSION == "train":
            pass
        elif SESSION == "test":
            self.model = self.load_model("models/rnn-0")
        if SESSION  == "test":
            if TEST_TYPE == 'image':
                pass
            elif TEST_TYPE == "webcam":
                self.process_web_cam()
    def arg_max(self,array):
        max_value = array[0]
        index = 0
        for i,el in enumerate(array):
            if el > max_value:
                index = i
                max_value = el
        return index
    def overlay(self,frame, rectangles, text, color=(48, 12, 160)):
        """
        Draw rectangles and text over image

        :param Mat frame: Image
        :param list rectangles: Coordinates of rectangles to draw
        :param list text: List of emotions to write
        :param tuple color: Box and text color
        :return: Most dominant emotion of each face
        :rtype: list
        """

        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(),rectangle.top()), (rectangle.right(),rectangle.bottom()), color)
            cv2.putText(frame, text[i], (rectangle.left() + 10, rectangle.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color)
        return frame
    def process_video(self,path):
        self.process_web_cam(path)
    def process_web_cam(self,path=-1):
        # model = model_from_json(open("models/rnn/rnn-0.json").read())
        # model.load_weights("models/rnn/rnn-0.h5")
        
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        face_detector = dlib.get_frontal_face_detector()
        currentEmotion = ""
        while cap.isOpened():
            
            sequences = np.zeros((self.max_sequence_length,self.input_shape[0],self.input_shape[1],self.input_shape[2]) )
            index = 0
            while index<self.max_sequence_length:
                ret, frame = cap.read()
                faces,rectangles = self.preprocessor.get_faces(frame,face_detector)
                if len(faces) >0:
                    face = faces[0]
                    face = self.preprocessor.sanitize(face)
                    face = face.astype(np.float32)/255
                    sequences[index] = face.reshape(-1,48,48,1)
                    index +=1
                    self.overlay(frame,[rectangles[0]],[currentEmotion])
                cv2.imshow("Image",frame)
                if (cv2.waitKey(10) & 0xFF == ord('q')):
                    break
            sequences = sequences.reshape(-1,self.max_sequence_length,48,48,1)
            predictions = self.model.predict(sequences)[0]
            print predictions
            emotion = self.arg_max(predictions)
            if predictions[emotion]>0.2:
                currentEmotion = self.preprocessor.classifier.get_string(emotion)    
            else:
                currentEmotion =""

            
        cv2.destroyAllWindows()
    def load_model(self,path):
        with open(path+".json") as jsonFile:
            model = model_from_json(jsonFile.read())
            model.load_weights(path+".h5")
            return model

        