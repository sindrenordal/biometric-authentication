import keras as k
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as r
import os
import zipfile

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from preprocessing.reformatting import find_subject_ids

# Load processed csv-file
def load_file(filename):
    data_location = "../Accelerometer_data/"+filename+".csv"
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), data_location))
    columns = ['X', 'Y', 'Z', 'SubjectID']
    return df.values

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

def load_dataframe(subject_ids):
    file_contents = []
    loaded_files = []
    for subject_id in subject_ids:
        print(subject_id)
        with zipfile.ZipFile("public_dataset/" + subject_id+".zip") as file:
            contents = []
            for file_name in file.namelist():
                if "Accelerometer" in file_name:
                    df = pd.read_csv(file.open(file_name))
                    df['SubjectID'] = subject_id
                    df.columns = ["SysTime", "EventTime", "ActivityID", "X", "Y", "Z", "Rotation", "SubjectID"]
                    keep = ["X","Y","Z","SubjectID", "ActivityID"]
                    df = df[keep]
                    content = df.values
                    loaded_files.extend(divide_windows(content, 1500))
    loaded = np.stack(loaded_files)
    return loaded

'''
Training size: 75%,
Test size: 25%
Number of subjects: 10

'''
def split(subject_id, data):
    print("split")

    X = data[:, :, :3]
    y = data[:, :, 3]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, random_state=20)
    y_train = y_train[:,0]
    y_test = y_test[:,0]
    y_train = (y_train == subject_id).astype(int)
    y_test = (y_test == subject_id).astype(int)

    return X_train, X_test, y_train, y_test
    

def prepare_data(users, binary):
    subject_ids = find_subject_ids("public_dataset/")
    subject_ids = subject_ids[21:23] #Reduced for development purposes
    df = load_dataframe(subject_ids)
    X_train, X_test, y_train, y_test = split(df, binary)
    return X_train, X_test, y_train, y_test   

def evaluate_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 25
    batch_size = 256
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    model = Sequential()
    model.add(CuDNNLSTM(64, return_sequences=True, input_shape = (n_timesteps, n_features)))
    #model.add(Dropout(0.1))
    model.add(CuDNNLSTM(64))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Plot accuracy
    plot = False
    if plot:
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

    print("Done fitting")
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

    # Confusion matrix
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)

    cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)

    return accuracy



if __name__ == "__main__":
    subject_ids = find_subject_ids("public_dataset/")
    #    subject_ids = subject_ids[21:31] #Reduced for development purposes
    for subject_id in subject_ids:
        # Choose random subjectIDS to create negative set
        duplicate = False
        while(not duplicate):
            duplicate = True
            ref1 = subject_ids[r.randint(0,len(subject_ids)-1)]
            ref2 = subject_ids[r.randint(0,len(subject_ids)-1)]
            if(subject_id == ref1 or subject_id == ref2 or ref1 == ref2):
                duplicate = True
            print(subject_id,ref1,ref2)
        df_pos = load_dataframe([subject_id])
        df_neg = load_dataframe([ref1, ref2])

        #Check for imbalance
        if(len(df_neg)/len(df_pos) > 1):
            df_neg = df_neg[:len(df_pos)]
            print("imbalance")

        X_train_pos, X_test_pos, y_train_pos, y_test_pos = split(subject_id, df_pos)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = split(subject_id, df_neg)
        X_train = np.concatenate([X_train_pos, X_train_neg], axis=0)
        X_test = np.concatenate([X_test_pos, X_test_neg], axis=0)
        y_train = np.concatenate([y_train_pos, y_train_neg], axis=0)
        y_test = np.concatenate([y_test_pos, y_test_neg], axis=0)
        print(y_train.shape)
        print(y_test.shape)
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
        acc = evaluate_model(X_train, X_test, y_train, y_test)
        print("Accuracy: ", acc)

