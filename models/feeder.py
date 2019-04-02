import pandas as pd
import numpy as np
import random as r
import matplotlib.pyplot as plt
import os
import zipfile
from preprocessing.reformatting import find_subject_ids

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Config to run on GPU


'''
1) Import data
2) Divide data into windows
3) Send data ta model
'''
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
                    divide_windows(content, 2000)
    loaded = np.stack(LOADED_FILES)
    return loaded

'''
Training size: 75%,
Test size: 25%
Number of subjects: 10

'''
def split(data):
    print("split")
    X = data[:, :, :3]
    y = data[:, :, 3]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=20)
    y_train = y_train[:,0]
    y_test = y_test[:,0]
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return X_train, X_test, y_train, y_test
    

def prepare_data():
    subject_ids = find_subject_ids("public_dataset/")
    subject_ids = subject_ids[:5] #Reduced for development purposes
    df = load_dataframe(subject_ids)
    X_train, X_test, y_train, y_test = split(df)
    return X_train, X_test, y_train, y_test   