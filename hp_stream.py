import streamlit as st
from PIL import Image
import numpy as np
import cv2, joblib
import keras
from keras.models import load_model
st.write("# Harry Potter Series Cast Face Recognition")
st.write("### This model can predict only some main lead characters of the series. However, the model is not that much accurate, so it will make errors.")
st.write("### Do not give image of any person other than the cast of HP series. ")
st.write("### Avoid giving image containing more than 1 person for good results.")
inp_image = st.file_uploader("Drop your image here")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
def get_cropped_image_if_2_eyes(img):
    if (img is not None):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes)>=2:
                return roi_color
            
model = load_model("hp.h5")
label = {'Alan Rickman': 0,
 'Daniel Radcliffe': 1,
 'Emma Watson': 2,
 'Gary Oldman': 3,
 'Helena Bonham Carter': 4,
 'Maggie Smith': 5,
 'Michael Gambon': 6,
 'Ralph Fiennes': 7,
 'Robbie Coltrane': 8,
 'Rupert Grint': 9,
 'Tom Felton': 10}

if inp_image is not None:
    read = Image.open(inp_image)
    arr = np.array(read)
    img = get_cropped_image_if_2_eyes(arr)

    if img is not None:
        img = np.array(img)
        resized_img = cv2.resize(img,(180,180))
        empty = np.zeros((100,180,180,3))
        empty[0]= resized_img
        empty= empty/255
        prediction = model.predict(empty)
        answer = np.argmax(prediction[0])
        for i,j in label.items():
            if j==answer:
                st.write("## ",i) 
    else:
        st.write("## :pensive: Error! try other image")



