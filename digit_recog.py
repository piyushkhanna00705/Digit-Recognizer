# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:44:05 2018

@author: Dell
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils 
#fixing random seed for reproducibility
seed=7
numpy.random.seed(seed)
#loading data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#flattening images
num_pixels=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test=X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

#normalise the inputs
X_train=X_train/255
X_test=X_test/255

#one-hot encoding outputs
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

#define baseline model
def baseline_model():
    #create model
    model=Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    #compile the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#build the model
model=baseline_model()
#fit the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=2)
#evaluation
scores=model.evaluate(X_test,y_test,verbose=0)
print("Baseline Error: %.2f%%"%(100-scores[1]*100))
    
    
