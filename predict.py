#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMG_SHAPE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)
parser.add_argument('--trained_network', default ='my_model.h5', help='trained model', type = str)
parser.add_argument('--top_k', default = 5, help = 'Top 5 classes.', type = int)
parser.add_argument('--category_names' , default = 'label_map.json', help = 'Categories', type = str)
commands = parser.parse_args()

image_path = commands.image_dir
keras_model = commands.trained_network
classes = commands.category_names
top_k = commands.top_k
loadmodel = tf.keras.models.load_model(keras_model,custom_objects={'KerasLayer': hub.KerasLayer})

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))/255.0
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)

    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))

    PROBS, CLASS = tf.math.top_k(prediction, top_k)
#dont forget that our json data frame starts at [0], so add 1!!
    TOP_CLASSES = [class_names[str(value+1)] for value in CLASS.cpu().numpy()[0]]

    print('Top classes, sorted by most accurate to the least', TOP_CLASSES)
    return PROBS.numpy()[0], TOP_CLASSES

probs, classes = predict(image_path, loadmodel, top_k)

