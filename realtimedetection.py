import cv2

from tensorflow.keras.models import model_from_json
import numpy as np
import streamlit as st
# from keras_preprocessing.image import load_img
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# webcam=cv2.VideoCapture(1)  #phone's camera --> droidcamp
webcam=cv2.VideoCapture(0)  #main camera --> laptop
# webcam=cv2.VideoCapture(2)  #main camera --> laptop
# webcam=cv2.VideoCapture(3)  #main camera --> laptop
# webcam=cv2.VideoCapture(4)  #main camera --> laptop
# webcam=cv2.VideoCapture(5)  #main camera --> laptop
# webcam=cv2.VideoCapture(6)  #main camera --> ob virtual camera

st.title('Minor of Shalini Tomar')
FRAME_WINDOW = st.image([])

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

if st.checkbox('Run the app'):
    while True:
        i,im=webcam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(im,1.3,5)
        try: 
            for (p,q,r,s) in faces:
                image = gray[q:q+s,p:p+r]
                cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
                image = cv2.resize(image,(48,48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                im_c = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                cv2.putText(im_c, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
                FRAME_WINDOW.image(im_c)
               
            # cv2.imshow("Output",im)
            cv2.waitKey(50)
        except cv2.error:
            pass



#streamlit run .\realtimedetection.py  --> run cmd







