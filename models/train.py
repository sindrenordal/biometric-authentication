#!/usr/bin/env python

import pandas as pd
import numpy as np
import random as r
import os
import zipfile

from tensorflow.python import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Dropout, LSTM

def find_subject_ids(dataset_name):
    subject_ids = []
    for file_name in os.listdir(dataset_name):
        subject_ids.append(file_name.split(".")[0])
    return subject_ids[:-1]

def train_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 50
    batch_size = 512
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    model.add(LSTM(25, input_shape=(n_timesteps, n_features)))
    model.add(Dense(25))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    print("Test accuracy: ", acc)
    return model, acc

def load_file(subjectid, type):
    X_name = ""
    if (type == "train"):
        X_name = "df_train.csv"
    elif (type == "test"):
        X_name = "df_test.csv"
    elif (type == "val"):
        X_name = "df_val.csv"
    X_location = "split/"+subjectid+"/"+X_name
    X = pd.read_csv(X_location)

def split(subject_id, data):
    X = data[:, :, :3]
    y = data[:, :, 3]

    y = y[:,0]
    y = (y == int(subject_id)).astype(int)
    return X, y

# Divide loaded data into windows of a given size
def divide_windows(data, window_size):
    shape = data.shape
    overflow = shape[0]%window_size
    # Assign startpoint as a random int between 0 and the overflow
    start_point = r.randint(0, overflow)
    number_windows = shape[0]//window_size #number of full windows
    windows = []
    for i in range(0, number_windows):
        start = start_point + window_size*i
        end = start_point + window_size*(i+1)
        windows.append(data[start:end, :])
    return windows

def load_dataframe(subject_id, type):
    loaded_files = []
    X_name = ""
    if (type == "train"):
        X_name = "df_train.csv"
    elif (type == "test"):
        X_name = "df_test.csv"
    elif (type == "val"):
        X_name = "df_val.csv"
    X_location = "split/" + subject_id + "/" + X_name
    df = pd.read_csv(X_location)
    content = df.values
    loaded_files = divide_windows(content, 300)
    loaded = np.stack(loaded_files)
    return loaded

def load_neg(subjectID, subject_ids, windows, type):
    loaded_files = []
    X_name = ""
    if (type == "train"):
        X_name = "df_train.csv"
    elif (type == "test"):
        X_name = "df_test.csv"
    elif (type == "val"):
        X_name = "df_val.csv"
    for i in range(0, windows):
        subject_id = subject_ids[r.randint(0, len(subject_ids)-1)]
        if(subject_id == subjectID):
            continue
        else:
            X_location = "split/" + subject_id + "/" + X_name
            df = pd.read_csv(X_location, nrows=300)
            content = df.values
            loaded_files.append(content)
    loaded = np.stack(loaded_files)
    return loaded


def run_training():
    subject_ids = find_subject_ids("split/")
    acc = []
    for subject_id in subject_ids:
        print(subject_id)
        df_pos_train = load_dataframe(subject_id, "train")
        df_pos_test = load_dataframe(subject_id, "test")
        length_train = df_pos_train.shape[0]
        length_test = df_pos_test.shape[0]
        df_train_neg = load_neg(subject_id, subject_ids, length_train, "train")
        df_test_neg = load_neg(subject_id, subject_ids, length_test, "test")


        # Handle imbalance
        if(len(df_train_neg)/len(df_pos_train) > 1):
            df_train_neg = df_train_neg[:len(df_pos_train)]
            df_test_neg = df_test_neg[:len(df_pos_test)]

        print(df_pos_train.shape)
        print(df_train_neg.shape)

        df_train = np.concatenate([df_pos_train, df_train_neg], axis=0)
        df_test = np.concatenate([df_pos_test, df_test_neg], axis=0)

        X_train, y_train = split(subject_id, df_train)
        X_test, y_test = split(subject_id, df_test)

        # Format for binary classification
        
        y_train = np.array(y_train)
        y_train = pd.DataFrame(data=y_train, columns=['y'])
        y_train = pd.get_dummies(y_train, columns=['y'])
        y_test = np.array(y_test)
        y_test = pd.DataFrame(data=y_test, columns=['y'])
        y_test = pd.get_dummies(y_test, columns=['y'])
        

        model, metric = train_model(X_train, X_test, y_train, y_test)
        acc.append(metric[1])
        model_path = "models/min_1600/"+subject_id+".h5"
        model.save(model_path)

if __name__ == "__main__":
    run_training()



