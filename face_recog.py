import cv2 as cv
import numpy as np


people = ['Siddharth']
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

faces_recognizer = cv.face.LBPHFaceRecognizer_create()

faces_recognizer.read('face_trained.yml')

vid = cv.VideoCapture(0)

while True:
    isTrue,frame = vid.read()
    # cv.imshow('faces',frame)
    grayframe = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(grayframe,1.1,4)

    for (x,y,w,h) in faces_rect:
        faces_roi = grayframe[y:y+h,x:x+w]

        label,confidence = faces_recognizer.predict(faces_roi)

        print(f'label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame,str(people[label]),(x+w,y+h),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    cv.imshow('Recog',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

vid.release()
cv.destroyAllWindows()


cv.waitKey(0)