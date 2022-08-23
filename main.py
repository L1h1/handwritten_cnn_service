import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import shutil
from fastapi import FastAPI,File,UploadFile


def process_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.bitwise_not(frame)
    frame = cv2.resize(frame, (28,28))
    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.reshape((28,28,1))
    frame = frame.astype('float32')/255.0
    frame =  np.expand_dims(frame, 0)
    return frame


def pred(image_path):
    image = process_image(image_path)
    res = model.predict(image)
    return res

model = tf.keras.models.load_model("checkpoint_bymerge_90_acc.h5",
                                     custom_objects={"KerasLayer":hub.KerasLayer})

dict_balanced=['0','1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','d','e','f','g','h','n','q','r','t']


app = FastAPI()

@app.post("/")
async def root(file: UploadFile = File(...)):
    with open('user_input.jpg','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"Estimated Symbol":dict_balanced[np.argmax(pred('user_input.jpg'))]}
