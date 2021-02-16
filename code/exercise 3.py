"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))

# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255

#make copies of the training, validation and test sets to prevent
#changing the original data
y_train_copy= y_train.copy()
y_val_copy= y_val.copy()
y_test_copy= y_test.copy()

#reduce to the 4 classes for y_train, y_val and y_test with a for loop
for i in range(len(y_train_copy)): 
    if (y_train_copy[i]==1 or y_train_copy[i]==7):
        y_train_copy[i]=0
        continue
    if (y_train_copy[i]==0 or y_train_copy[i]==6 or y_train_copy[i]==8 or y_train_copy[i]==9):
        y_train_copy[i]=1
        continue
    if (y_train_copy[i]==2 or y_train_copy[i]==5):
        y_train_copy[i]=2
        continue
    if (y_train_copy[i]==3 or y_train_copy[i]==4):
        y_train_copy[i]=3
        continue
    
for i in range(len(y_val_copy)): 
    if (y_val_copy[i]==1 or y_val_copy[i]==7):
        y_val_copy[i]=0
        continue
    if (y_val_copy[i]==0 or y_val_copy[i]==6 or y_val_copy[i]==8 or y_val_copy[i]==9):
        y_val_copy[i]=1
        continue
    if (y_val_copy[i]==2 or y_val_copy[i]==5):
        y_val_copy[i]=2
        continue
    if (y_val_copy[i]==3 or y_val_copy[i]==4):
        y_val_copy[i]=3
        continue
    
for i in range(len(y_test_copy)): 
    if (y_test_copy[i]==1 or y_test_copy[i]==7):
        y_test_copy[i]=0
        continue
    if (y_test_copy[i]==0 or y_test_copy[i]==6 or y_test_copy[i]==8 or y_test_copy[i]==9):
        y_test_copy[i]=1
        continue
    if (y_test_copy[i]==2 or y_test_copy[i]==5):
        y_test_copy[i]=2
        continue
    if (y_test_copy[i]==3 or y_test_copy[i]==4):
        y_test_copy[i]=3
        continue

# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train_copy, 4)
y_val = to_categorical(y_val_copy, 4)
y_test = to_categorical(y_test_copy, 4)


model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# two layers both with 256 neurons and ReLU nonlinearity (see exercise 1 and 2)
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
# output layer with 4 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax')) 


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="layer2neuron256"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])


score = model.evaluate(X_test, y_test, verbose=0)


print("Loss: ",score[0])
print("Accuracy: ",score[1])


