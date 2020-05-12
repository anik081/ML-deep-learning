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
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
#data directortory
DATADIR = "D:\Image_Processing\plant_diseases\color"
#name of classes
CATEGORIES = ["Apple___Black_rot",
              "Apple___healthy",
              "Apple___Apple_scab",
              "Apple___Cedar_apple_rust",
          ]
#print number of classes
noOfClasses = len(CATEGORIES)
print('No of classes:%d' %noOfClasses)

#load image
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  #path to classes
    for img in os.listdir(path):
        #for gray IMREAD_GRAYSCALE
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        plt.imshow(img_array)
        plt.show()
        break
    break
print(img_array.shape)

#resizing image . in this case 50 by 50
IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
#show the resized image
plt.imshow(new_array)
plt.show()

#create training data
training_data =[]
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)  #path to classes
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #for gray IMREAD_GRAYSCALE
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                IMG_SIZE = 50
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
create_training_data()
print(len(training_data))

#shuffle all images to avoid overfit
import random
random.shuffle(training_data)
"""
for sample in training_data[:10]:
    print(sample[1])
"""
#seperate class and features
X = []
y = []
for features, label in training_data:
        X.append(features)
        y.append(label)
#convert image into numpy array to rehsape      
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3) #for coloured  3 , for gray it will be 1 . here (3) 
#one hot encode to class
y=  to_categorical(y,noOfClasses)
#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X)

#save tensorboard logs
NAME ="Cats-Dogs-cnn-64-{}".format(int(time.time()))
tensorboard = TensorBoard (log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
))
#checkpoint to check the heightest accuracy and save that epoch
checkpointer = ModelCheckpoint(filepath="best_weights.model", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

#before feeding data we need to normalize(keras.utils.normalize-- most probably) but 
#as image data are min 0 and max 255(pixel data)
X = X/255.0

print(X.shape)


















