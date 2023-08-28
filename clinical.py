import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
from xml.dom import minidom
import base64
import random
import re
import cv2
import seaborn as sns
import streamlit.components.v1 as html
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from scipy import stats
from streamlit_option_menu import option_menu
from PIL import Image
from skimage import transform

class_names = ['Acne and Rosacea Photos',
 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
 'Atopic Dermatitis Photos',
 'Bullous Disease Photos',
 'Cellulitis Impetigo and other Bacterial Infections',
 'Eczema Photos',
 'Exanthems and Drug Eruptions',
 'Hair Loss Photos Alopecia and other Hair Diseases',
 'Herpes HPV and other STDs Photos',
 'Light Diseases and Disorders of Pigmentation',
 'Lupus and other Connective Tissue diseases',
 'Melanoma Skin Cancer Nevi and Moles',
 'Nail Fungus and other Nail Disease',
 'Poison Ivy Photos and other Contact Dermatitis',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Scabies Lyme Disease and other Infestations and Bites',
 'Seborrheic Keratoses and other Benign Tumors',
 'Systemic Disease',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Urticaria Hives',
 'Vascular Tumors',
 'Vasculitis Photos',
 'Warts Molluscum and other Viral Infections']

BATCH_SIZE = 64
IMAGE_SIZE = 128
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

resize_and_rescale = tf.keras.Sequential([
                     layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
                     layers.experimental.preprocessing.Rescaling(1.0/255)
])

base_model.trainable = False
model = tf.keras.Sequential([
    resize_and_rescale,
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'), 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(23,activation='softmax')
])

model.build(input_shape = (BATCH_SIZE,128,128,3))

model.load_weights("dermnet.h5")

def predict(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')  ##/255
#     print(np_image)
    np_image = transform.resize(np_image, (128, 128, 3))
    np_image = np.expand_dims(np_image, axis=0)
    
    predictions = model.predict(np_image)
    print(predictions)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


st.sidebar.title("ERHA AI Clinical")

uploaded_file = st.sidebar.file_uploader("Choose image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.info("Image Upload Successfully...")
    st.image(opencv_image, width=500 ,channels="BGR")

    st.success("Prediction is {} with confidence {}%".format(predict(uploaded_file)[0],
                                                            predict(uploaded_file)[1]))