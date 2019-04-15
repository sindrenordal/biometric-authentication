import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

from feeder import prepare_data

'''
Information from: "Smartphone Continuous Authentication Using Deep Learning Autoencoders"
Window-size = 500
Input-dim = 500 * 3
Number of hidden units each layer = 1500
Number of layers = 5
'''

#Reformat to binary output
def reformat(y):
    for x in range(0, len(y)):
        if(y[x] == '100669'):
            y[x] = 1;
        else:
            y[x] = 0

def evaluate_model(X_train, X_test, y_train, y_test):
    time_periods, num_features = X_train.shape[1], X_train.shape[2]
    input_shape = (time_periods * num_features)
    n_outputs = y_train.shape[1]

    X_train = X_train.reshape(X_train.shape[0], input_shape)
    X_test = X_test.reshape(X_test.shape[0], input_shape)
    print(X_train)

    verbose = 1
    epochs = 50
    batch_size = 64

    model = Sequential()
    model.add(Dense(250, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    print("Done fitting")
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(2, False)
    #y_train = reformat(y_train)
    #y_test = reformat(y_test)
    acc = evaluate_model(X_train, X_test, y_train, y_test)
    print(acc)

