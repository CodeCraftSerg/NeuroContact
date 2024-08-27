import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import keras
import gdown

from os.path import exists
from PIL import Image


if not exists("./app_predict/model/cifar10_result.keras"):
    url = "https://drive.google.com/file/d/1iQZbLZZAAQP3CdqxIi1PWE73xv7fQxqq/view?usp=sharing"
    output = "cifar10_result.keras"
    gdown.download(url, output, quiet=False, fuzzy=True)

model_filename = "./app_predict/model/cifar10_result.keras"
model = keras.models.load_model(model_filename)


LABEL_NAMES = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def resize_image(image_to_recognize):
    img = Image.open(image_to_recognize)
    img = img.resize((32, 32))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict_image(image):
    input_img = resize_image(image)
    predict_img = model.predict(input_img)
    pred_class = LABEL_NAMES[np.argmax(predict_img)]
    return pred_class
