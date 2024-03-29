import os
import cv2 as cv
import numpy as np

people = ['Siddharth']
DIR = r'C:\OpenCV\faces'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []

def create_train():
    label = 0
    for img in os.listdir(DIR):
        img_path = os.path.join(DIR,img)

        img_array = cv.imread(img_path)
        gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h,x:x+w]
            features.append(faces_roi)
            labels.append(label)

create_train()
print("Training Done--------------")
features = np.array(features,dtype='object')
labels = np.array(labels)
faces_recognizer = cv.face.LBPHFaceRecognizer_create()

faces_recognizer.train(features,labels)

faces_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)