import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

__class_name_number = {}
__class_number_name = {}

__model = None

def classify_img(img_b64,file_path = None):
    result = []
    imgs = get_cropped_img(file_path,img_b64)

    for img in imgs:
        scale_raw = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scale_har = cv2.resize(img_har,(32,32))
        comb_img = np.vstack((scale_raw.reshape((32*32*3),1),scale_har.reshape(32*32,1)))

        len_image = 32*32*3 + 32*32

        final = comb_img.reshape(1,len_image).astype(float)
        result.append({
            'class' : classno_name(__model.predict(final)[0]),
            'class_probability': np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_number})


    return result
         

def load_saved_artifacts():
    print('loading saved artifacts')
    global __class_name_number
    global __class_number_name
     
    with open('./Server/Artifacts/class_dictionary.json','r') as f:
        __class_name_number = json.load(f)
        __class_number_name = {v:k for k,v in __class_name_number.items()}
    
    global __model
    if __model is None:
        with open('./Server/Artifacts/saved_model.pkl','rb') as f:
            __model = joblib.load(f)
    
    print('loading saved artifacts...done')



def get_cv2_from_b64(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data),np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_img(image_path, image_b64_data):
    # Same as Notebook
    face_cascade = cv2.CascadeClassifier('/Users/alpha/Downloads/CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/alpha/Downloads/CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_eye.xml')
    
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_from_b64(image_b64_data)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    cropped_img = []

    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cropped_img.append(roi_color)

    return cropped_img
        

def classno_name(num):
    return __class_number_name[num]

def get_b64():
    with open('./Server/Program/b64.txt') as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
