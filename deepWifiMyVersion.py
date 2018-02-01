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
    for i in range(num_to_remove):
        d.pop()
    d.add(Dense(128, input_dim=64, activation='tanh', bias=True))
    d.add(Dense(128, activation='tanh', bias=True))
    d.add(Dense(num_classes, activation='softmax', bias=True))
    d.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return d



path_train = "radioMap.csv"
path_validation = "wifi_4th_floor_test.csv"

#Explicitly pass header=0 to be able to replace existing names 
train_df = pd.read_csv(path_train,header = 0)
train_AP_strengths = train_df.ix[:,:185] #select first 520 columns

#Scale transforms data to center to the mean and component wise scale to unit variance
train_AP_features = scale(np.asarray(train_AP_strengths))
train_labels = np.asarray(train_df.ix[:,185])
#train_labels_encoding = zeros((98, 98))
train_labels = to_categorical(train_labels)




nb_epochs = 20
batch_size = 1
input_size = 185
num_classes = 98

e = encoder()
d = decoder(e)
d.fit(train_AP_features, train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
# c = classifier(d)
# c.fit(train_AP_features, train_labels, nb_epoch=nb_epochs, batch_size=batch_size)
# loss, acc = c.evaluate(test_AP_features, test_labels)
# print (loss, acc)
#result = c.predict(val_X)
#result_new = np.argmax(result, axis=1)
#print (result_new[:10])
#print (np.argmax(val_y, axis=1)[:10])