# import cv2
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
    return "ESTO ES LO QUE TIENE QUE SERVIR"



# app.post('/img', (req, res) => {
#   res.header("Content-Type", "application/json");
#   console.log(req.body)
#   res.send(JSON.stringify({
#     status: 200,
#     emotion: 'happy'
#   }))
# })


#request.get_data()

@app.route('/img', methods=['POST'])
def parse_request():
    
    data = request.data
    p = np.array(data)
    if (data == "yes"):
        return str("YES")
    elif (data == 2):
        return str("NO")
    else:
        return str(p)
    
    # empty in some cases
    # always need raw data here, not parsed form data

app.run(debug=True)




def predict_img(dir):
    img = dir
    new_img = cv2.resiz(img, (48,48))
    new_img = np.expand_dims(new_img, axis=0)
    new_img = new_img/255.0

    predictions = model.predict(new_img)

    if (np.argmax(predictions, axis=1) == 0):
        return str("Angry")
    elif (np.argmax(predictions, axis=1) == 1):
        return str("Disgust")
    elif (np.argmax(predictions, axis=1) == 2):
        return str("Fear")
    elif (np.argmax(predictions, axis=1) == 3):
        return str("Happy")
    elif (np.argmax(predictions, axis=1) == 4):
        return str("Neutral")
    elif (np.argmax(predictions, axis=1) == 5):
        return str("Sad")
    else:
        return str("Surprise")





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

