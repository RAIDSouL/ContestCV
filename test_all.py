from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

IMAGE_SIZE = (256,256)

def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

model = load_model('Models/256/my_model_epoch75.h5',custom_objects={'iou':iou})

for i in os.listdir('textlocalize/validation/Input/'):
    test_im = cv2.imread('textlocalize/validation/Input/'+str(i))
    true_size = test_im.shape
    imshow_size = (512,round(true_size[0]*512/true_size[1]))
    cv2.imshow('Input',cv2.resize(test_im, imshow_size))
    # cv2.waitKey(50)

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    #segmented = np.around(segmented)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')

    cv2.imshow('Output',cv2.resize(segmented, imshow_size))
    cv2.waitKey()

    # old
    # cv2.imwrite('Answer/'+str(i),cv2.resize(segmented, imshow_size)) 
    # new
    cv2.imwrite('Answer/'+str(i),cv2.resize(segmented,(true_size[1],true_size[0])))
