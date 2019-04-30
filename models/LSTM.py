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

LOADED_FILES = []

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
        LOADED_FILES.append(data[start:end, :])

def load_dataframe(subject_ids):
    file_contents = []
    loaded_files = list()
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
                    divide_windows(content, 1000)
    loaded = np.stack(LOADED_FILES)
    return loaded

'''
Training size: 75%,
Test size: 25%
Number of subjects: 10

'''
def split(subject_id, data):
    print("split")

    #data[:,:,0] = (data[:,:,0] - data[:,:,0].min())/(data[:,:,0].max() - data[:,:,0].min())
    #data[:,:,1] = (data[:,:,1] - data[:,:,1].min())/(data[:,:,1].max() - data[:,:,1].min())
    #data[:,:,2] = (data[:,:,2] - data[:,:,2].min())/(data[:,:,2].max() - data[:,:,2].min())
    X = data[:, :, :3]
    y = data[:, :, 3]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=20)
    y_train = y_train[:,0]
    y_test = y_test[:,0]
    y_train = (y_train == subject_id).astype(int)
    y_test = (y_test == subject_id).astype(int)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return X_train, X_test, y_train, y_test
    

def prepare_data(users, binary):
    subject_ids = find_subject_ids("public_dataset/")
    subject_ids = subject_ids[21:23] #Reduced for development purposes
    df = load_dataframe(subject_ids)
    X_train, X_test, y_train, y_test = split(df, binary)
    return X_train, X_test, y_train, y_test   

def evaluate_model(X_train, X_test, y_train, y_test):
    verbose = 1
    epochs = 15
    batch_size = 256
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    print(n_outputs)
    model = Sequential()
    model.add(CuDNNLSTM(64, return_sequences=True, input_shape = (n_timesteps, n_features)))
    model.add(CuDNNLSTM(64))
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
    subject_ids = find_subject_ids("public_dataset/")
    #    subject_ids = subject_ids[21:31] #Reduced for development purposes
    for subject_id in subject_ids:
        # Choose random subjectIDS to create negative set
        duplicate = False
        while(not duplicate):
            duplicate = True
            ref1 = subject_ids[r.randint(0,len(subject_ids))]
            ref2 = subject_ids[r.randint(0,len(subject_ids))]
            ref3 = subject_ids[r.randint(0,len(subject_ids))]
            if(subject_id == ref1 or subject_id == ref2 or subject_id == ref3):
                duplicate = True
            print(subject_id,ref1,ref2,ref3)
        df = load_dataframe([subject_id, ref1, ref2, ref3])
        X_train, X_test, y_train, y_test = split(subject_id, df)
        print(y_train.shape)
        acc = evaluate_model(X_train, X_test, y_train, y_test)
        print("Accuracy: ", acc)

