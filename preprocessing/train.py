#!/usr/bin/env python

import pandas as pd
import numpy as np
import random as r
import os
import zipfile

from tensorflow.python import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Dropout

def find_subject_ids(dataset_name):
    subject_ids = []
    for file_name in os.listdir(dataset_name):
        subject_ids.append(file_name.split(".")[0])
    return subject_ids[:-1]

def train_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 25
    batch_size = 256
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    model.add(CuDNNLSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(CuDNNLSTM(64))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    print("Test accuracy: ", acc)
    return model

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
    loaded_files = divide_windows(content, 1500)
    loaded = np.stack(loaded_files)
    return loaded

def run_training():
    subject_ids = find_subject_ids("split/")
    for subject_id in subject_ids:
        duplicate = True
        while(duplicate):
            duplicate = False
            ref1 = subject_ids[r.randint(0, len(subject_ids)-1)]
            ref2 = subject_ids[r.randint(0, len(subject_ids)-1)]
            if (subject_id == ref1 or subject_id == ref2 or ref1 == ref2):
                duplicate = True
            print(subject_id, ref1, ref2)
        df_pos_train = load_dataframe(subject_id, "train")
        df_pos_test = load_dataframe(subject_id, "test")
        df_neg1_train = load_dataframe(ref1, "train")
        df_neg1_test = load_dataframe(ref1, "test")
        df_neg2_train= load_dataframe(ref2, "train")
        df_neg2_test = load_dataframe(ref2, "test")

        # Concatenate negatives
        df_train_neg = np.concatenate([df_neg1_train, df_neg2_train], axis=0)
        df_test_neg = np.concatenate([df_neg1_test, df_neg2_test], axis=0)

        # Handle imbalance
        if(len(df_train_neg)/len(df_pos_train) > 1):
            df_train_neg = df_train_neg[:len(df_pos_train)]
            df_test_neg = df_test_neg[:len(df_pos_test)]

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

        model = train_model(X_train, X_test, y_train, y_test)
        model_path = "models/subject_id"+"neg"+ref1+"_"+ref2+".h5"
        model.save(model_path)

run_training()



