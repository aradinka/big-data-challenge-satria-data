from PIL import Image
import numpy as np

import tensorflow as tf
from keras import backend as K


def crop_face(photo_path, face_locations):
    im = Image.open(photo_path)
    left = face_locations[0][3]
    top = face_locations[0][0]
    right = face_locations[0][1]
    bottom = face_locations[0][2]

    lebar = right - left
    pelebar = lebar * 50/100
    left = left - pelebar
    top = top - pelebar
    right = right + pelebar
    bottom = bottom + pelebar

    im1 = im.crop((left, top, right, bottom))
    return im1

def biggest_face(photo_path, face_locations):
    """Detect biggest face"""
    save_im = []
    for face in face_locations:
        im = Image.open(photo_path)
        left = face[3]
        top = face[0]
        right = face[1]
        bottom = face[2]
        panjang = bottom-top
        lebar = right-left
        luas = panjang *lebar
        save_im.append(luas)
    face = save_im.index(max(save_im))
    return face


def time_decay(epoch, initial_lrate):
    decay_rate = 0.01
    new_lrate = initial_lrate/(1+decay_rate*epoch)
    return new_lrate

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))