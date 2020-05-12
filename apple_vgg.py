# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:27:51 2020

@author: Anik
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
DATADIR = "D:\Image_Processing\plant_diseases\color"

CATEGORIES = ["Apple___Black_rot",
              "Apple___healthy",
              "Apple___Apple_scab",
              "Apple___Cedar_apple_rust",
]

noOfClasses = len(CATEGORIES)
print('No of classes:%d' %noOfClasses)


for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  #path to cats or dogs
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        plt.imshow(img_array)
        plt.show()
        break
    break
print(img_array.shape)


IMG_SIZE = 224
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)
plt.show()
print(new_array.shape)

training_data =[]
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)  #path to cats or dogs
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
create_training_data()
len_train_data =len(training_data) 
print('total data : %d' %len_train_data)

import random
random.shuffle(training_data)
"""
for sample in training_data[:10]:
    print(sample[1])
"""
X = []
y = []
for features, label in training_data:
        X.append(features)
        y.append(label)
        
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3) #for coloured 1 will be replaced with 3
y=  to_categorical(y,noOfClasses)


NAME ="Cats-Dogs-cnn-64-{}".format(int(time.time()))
tensorboard = TensorBoard (log_dir = os.path.join(
    "vgg_logs_apple",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
))
#before feeding data we need to normalize(keras.utils.normalize-- most probably) but 
#as image data are min 0 and max 255(pixel data)
X = X/255.0

model = Sequential()

vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

model = Sequential()

for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
model.summary()

model.add(Dense(noOfClasses))
model.add(Activation("softmax"))

model.summary()


model.compile(loss="categorical_crossentropy",
             optimizer ="adam",
             metrics =['accuracy'])
model.fit(X,y, batch_size = 32, epochs =10 , validation_split = 0.1, callbacks =[tensorboard])
