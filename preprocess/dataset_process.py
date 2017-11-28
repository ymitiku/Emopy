from __future__ import print_function
import os
import numpy as np
import cv2
from constants import IMG_SIZE,EMOTIONS


def sanitize(image):
    if image is None:
        raise "Unable to sanitize None image"
    image = cv2.resize(image,IMG_SIZE)
    if len(image.shape)>2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image
def string2emotion(string):
    for emotion in EMOTIONS:
        if EMOTIONS[emotion] == string:
            return emotion
    raise Exception("Could not found emotion "+str(string))
def load_dataset(directory,verbose=False):
    if verbose:
        print("Loading dataset from ",directory, "dir")
    images = []
    emotions = []
    for em_dir in os.listdir(directory):
        for img_file in os.listdir(os.path.join(directory,em_dir)):
            img = cv2.imread(os.path.join(directory,em_dir,img_file))
            img = sanitize(img)
            images.append(img)
            emotions.append(string2emotion(em_dir))
        if verbose:
            print("Loadded",em_dir,"dataset")
    if verbose:
        print("Finished loading dataset")
    x = np.array(images)
    y = np.array(emotions)
    return x,y

