import keras as k
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from feeder import prepare_data
from tensorflow.python.client import device_lib


print(device_lib.list_local_devices())

def evaluate_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 25
    batch_size = 256
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    print(n_outputs)
    model = Sequential()
    model.add(CuDNNLSTM(64, return_sequences=True, input_shape = (n_timesteps, n_features)))
    #model.add(CuDNNLSTM(50, return_sequences=True, input_shape = (n_timesteps, n_features)))
    #model.add(CuDNNLSTM(50, return_sequences=True, input_shape = (n_timesteps, n_features)))
    #model.add(CuDNNLSTM(50, return_sequences=True, input_shape = (n_timesteps, n_features)))
    model.add(CuDNNLSTM(64))
    #model.add(Dropout(0.4))
    #model.add(Dense(50, activation='tanh'))
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

    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)

    return accuracy



if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = prepare_data(5, False)
    print(y_test.shape)
    acc = evaluate_model(X_train, X_test, y_train, y_test)
    print("Accuracy: ", acc)

