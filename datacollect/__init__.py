from __future__ import print_function

import dlib
import cv2
import os

detector  = dlib.get_frontal_face_detector()

def save_face(frame,face,file_path):
    face_image = frame[
                max(face.top()-100,0):min(face.bottom()+100,frame.shape[0]),
                max(face.left()-100,0):min(face.right()+100,frame.shape[1])

    ]
    if face_image.shape[0]>100:
        cv2.imwrite(file_path,face_image)

def collect_faces(video_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video",video_path)
        exit(0)
    print ("Processing video")
    number_of_faces = 0
    saved_faces = 0
    while cap.isOpened():
        ret,frame = cap.read()
        faces = detector(frame)
        for face in faces:
            number_of_faces += 1
            if number_of_faces%5 == 0:
                saved_faces += 1
                save_face(frame,face,os.path.join(output_path,str(saved_faces)+".png"))
                print ("collected ",saved_faces,"faces")