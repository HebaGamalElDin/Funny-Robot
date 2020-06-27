# -*- coding: utf-8 -*-
"""
Created on Tue May  5 07:22:01 2020

@author: Heba Gamal El-Din
"""

######################################
""" Importing Necessary Libraries """
######################################
import numpy as np
from keras.models import load_model
import cv2
from time import sleep
###################################
""" Models and Emotion Classes """
###################################
Emotions_Dic = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
Emotion_Model = load_model("model_v6_23.hdf5")
Face_Model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

################################
""" Face Detection Function """
###############################
def Face_Detect(Frame):
    ROIs = []
    Boxes = []
    Gray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
    Faces = Face_Model.detectMultiScale(Gray, 1.3, 5)
    for I in range(0, len(Faces)):
        (x,y) = (int(Faces[I][0]), int(Faces[I][1]))
        (w,h) = (int(Faces[I][2]), int(Faces[I][3]))
        Box = [(x,y),(x+w,y+h),(round(x+w*0.55),y-25)]
        Boxes.append(Box)
        ROI = Gray[y:y+h, x:x+w]
        try:
            ROI = cv2.resize(ROI, (48, 48), interpolation = cv2.INTER_AREA)
        except:
            ROI = ROI
        ROIs.append(ROI)
    return ROIs, Boxes

###################################
""" Emotion Detection Function """
##################################
def Emotion_Detect(Faces):
    Emotions = [None] * len(Faces)
    for indx, Face in enumerate(Faces):
        Face = Face.astype("float") / 255.0
        Face = np.asarray(Face)
        Face = Face.reshape(Face.shape[0], Face.shape[1], 1)
        Face = np.expand_dims(Face, axis=0)  
        Prdcts = Emotion_Model.predict(Face)
        Dict = dict((v,k) for k,v in Emotions_Dic.items())
        Emotion = Dict[Prdcts.argmax()]
        Emotions[indx] = Emotion
    return Emotions

######################################
""" Boxes Drawing and Text Posing """
#####################################
def Draw_Text(Frame,Boxes,Texts):
    for I in range(len(Boxes)):
        cv2.rectangle(Frame, Boxes[I][0], Boxes[I][1], (0,0,255),1)
        cv2.rectangle(Frame, (Boxes[I][0][0],Boxes[I][0][1]), Boxes[I][2], (0,0,255), cv2.FILLED)
        cv2.putText(Frame, Texts[I], (Boxes[I][0][0],Boxes[I][0][1]-5) , cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    return Frame
Cap = cv2.VideoCapture(0)
sleep(2.0)
FPS = Cap.get(cv2.CAP_PROP_FPS)
KPS = 5
hop = round(FPS / KPS)
curr_frame = 0
while True:
    Bool, Frame = Cap.read()
    if not Bool:
        break
    elif curr_frame % hop == 0:
        ROIs, Boxes = Face_Detect(Frame)
        Emotions = Emotion_Detect(ROIs)
        Frame = Draw_Text(Frame, Boxes, Emotions)
        cv2.imshow("Detectinf Facial Emotions", Frame)
        if cv2.waitKey(1) == 27:
            break
            
Cap.release()
cv2.destroyAllWindows()