import base64
import io
from imageio import imread
import cv2
import numpy as np
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(48, 48,3)))
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(7, activation="softmax"))

    model.load_weights('moodly_V4_Keras_Sequential_No_Hot_final.h5')

    return model

model = load_model()

# predicting feeling of img
def predict_img(img):
    new_img = cv2.resize(img, (48,48))
    new_img = np.expand_dims(new_img, axis=0)
    new_img = new_img/255.0

    predictions = model.predict(new_img)

    if (np.argmax(predictions, axis=1) == 0):
        return str("Angry")
    elif (np.argmax(predictions, axis=1) == 1):
        return str("Disgusted")
    elif (np.argmax(predictions, axis=1) == 2):
        return str("Fear")
    elif (np.argmax(predictions, axis=1) == 3):
        return str("Happy")
    elif (np.argmax(predictions, axis=1) == 4):
        return str("Neutral")
    elif (np.argmax(predictions, axis=1) == 5):
        return str("Sad")
    else:
        return str("Surprised")

@app.route('/')
def home():
    return "<h1>Moodly API</h1><p>The API is listening is listening for an image in '/img'</p>"

@app.route('/img', methods=['POST'])
def parse_request():
    # reading the json file
    try:
        obj = request.json
        print('calculating feeling')
        # convert base64 string to an numpy array
        img = imread(io.BytesIO(base64.b64decode(obj['img'])))

        # calculate the emotion of the image
        emotion = predict_img(img)
        print('feeling found: ', emotion)

        return jsonify({
            "status": 200,
            "emotion": emotion
        })
    except:
        return jsonify({
            "message": "Something went wrong in the API"
        })
    

app.run(host='0.0.0.0')