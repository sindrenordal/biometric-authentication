import keras as k
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from feeder import prepare_data

def evaluate_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 3
    batch_size = 32
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape = (n_timesteps, n_features)))
    model.add(LSTM(200))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print("Done fitting")
    print(model.summary())
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    acc = evaluate_model(X_train, X_test, y_train, y_test)
    print(acc)
