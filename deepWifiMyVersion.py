 # -*- coding: utf-8 -*-     

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


def encoder():
    model = Sequential()
    model.add(Dense(256, input_dim=input_size, activation='tanh', bias=True))
    model.add(Dense(128, activation='tanh', bias=True))
    model.add(Dense(64, activation='tanh', bias=True))
    return model
	
def decoder(e):   
    e.add(Dense(128, input_dim=64, activation='tanh', bias=True))
    e.add(Dense(256, activation='tanh', bias=True))
    e.add(Dense(input_size, activation='tanh', bias=True))
    e.compile(optimizer='adam', loss='mse')
    return e
	
def classifier(d):
    num_to_remove = 3
    #这里是为了将decoder的部分去掉，因为训练的auto-encoder只需要降维度的那个部分
    for i in range(num_to_remove):
        d.pop()
    d.add(Dense(128, input_dim=64, activation='tanh', bias=True))
    d.add(Dense(128, activation='tanh', bias=True))
    d.add(Dense(num_classes, activation='softmax', bias=True))
    d.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return d



path_train = "map.csv"
path_validation = "radioMap6.csv"

#以下的部分是用来生成ground truth的
#Explicitly pass header=0 to be able to replace existing names 
train_df = pd.read_csv(path_train,header = 0)
train_AP_strengths = train_df.ix[:,:414] #select first 520 columns

#Scale transforms data to center to the mean and component wise scale to unit variance
train_AP_features = scale(np.asarray(train_AP_strengths))
train_labels = np.asarray(train_df.ix[:,416])
#train_labels_encoding = zeros((98, 98))
train_labels = to_categorical(train_labels)

#一下生成我们需要的测试data吧
test_df = pd.read_csv(path_validation, header = 0)
test_AP_strengths = test_df.ix[:,:414] #select first 520 columns

#Scale transforms data to center to the mean and component wise scale to unit variance
test_AP_features = scale(np.asarray(test_AP_strengths))



nb_epochs = 20
batch_size = 1
input_size = 414
num_classes = 289

e = encoder()
d = decoder(e)
d.fit(train_AP_features, train_AP_features, nb_epoch=nb_epochs, batch_size=batch_size)
c = classifier(d)
c.fit(train_AP_features, train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
loss, acc = c.evaluate(train_AP_features, train_labels)
print (loss, acc)
result = c.predict(test_AP_features)
result_new = np.argmax(result, axis=1)
print (result_new)
#print (np.argmax(val_y, axis=1)[:10])