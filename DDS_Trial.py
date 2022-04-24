import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer



def gen_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = load_model(r'F:\Drowsiness Detection System\Trained_models\trained_model.h5')

    # Loading Alarm
    mixer.init()
    sound= mixer.Sound(r'F:\Drowsiness Detection System\alarm.wav')

    cap = cv2.VideoCapture(0)     #to access webcam

    #Checking if webcam is available or not
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOSError("Cannot open Webcam")
    score=0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            height,width = frame.shape[0:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     #converting colored image onto gray scale
            faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2,minNeighbors=1)
            eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
            
            cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
            cv2.rectangle(frame, (0,height-500),(400,height-450),(0,0,0),thickness=cv2.FILLED)
            cv2.putText(frame,"Press 'q' to stop the detection process. ",(0,height-460),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale=0.7,color=(255,0,0), thickness=1,lineType=cv2.LINE_AA)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=4 )
                    
            for (a,b,e_w,e_h) in eyes:
                
                # PREPROCESSING STEPS
                eye= frame[b:b+e_h,a:a+e_w]
                eye= cv2.resize(eye,(80,80))
                eye= eye/255
                eye= eye.reshape(80,80,3)
                eye= np.expand_dims(eye,axis=0)
                
                # preprocessing is done now model prediction
                pred = model.predict(eye)
                
                # when person is drowsy
                if pred[0][0]>0.30:
                    cv2.putText(frame,'Drowsy',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=0.8,color=(255,255,255),
                            thickness=1,lineType=cv2.LINE_AA)
                    cv2.putText(frame,'Score '+str(score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=0.7,
                                color=(255,255,255), thickness=1,lineType=cv2.LINE_AA)
                    
                    
                    score=score+1
                    if(score>25):
                        try:
                            sound.play()
                        except:
                            pass   
                # when person is not drowsy    
                if pred[0][1]>0.90:
                    score=score-1
                    if(score<0 or score>40):
                        score=0
                
            cv2.imshow('frame',frame)
            if cv2.waitKey(33) & 0xFF==ord('q'):   #to halt the process 'q' key needs to be pressed
                break

        
        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()

