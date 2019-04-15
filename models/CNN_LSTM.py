import keras as k
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, CuDNNLSTM

from feeder import prepare_data

def evaluate_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 3
    batch_size = 128
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    n_steps = 20
    n_length = 100
    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    model.add(TimeDistributed(Flatten()))
    model.add(CuDNNLSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print("Done fitting")
    print(model.summary())
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(5, False)
    acc = evaluate_model(X_train, X_test, y_train, y_test)
    print(acc)
