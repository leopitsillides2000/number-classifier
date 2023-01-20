from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten
import numpy as np


digits = load_digits()

def example(d):
    print (d.target[0], d.data[0])

example(digits)

def image_example(d):
    plt.gray()
    plt.matshow(d.images[0])
    plt.show()
    return

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], 8, 8, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 8, 8, 1))


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



def applyCNN(X_train):

    input_shape = (8,8,1)
    num_classes = 10

    model = Sequential()
    ## Getting dimensional error with Conv2D
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape = input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape = input_shape))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

model = applyCNN(X_train)

history = model.fit(X_train, y_train, epochs=50)
loss, accuracy = model.evaluate(X_test, y_test)