import pandas as pd
import numpy as np
import random as r
import os
import zipfile
import errno

from sklearn.model_selection import train_test_split

# Find the subjectIDs in the dataset
def find_subject_ids(dataset_name):
    subject_ids = []
    for file_name in os.listdir(dataset_name):
        subject_ids.append(file_name.split(".")[0])
    return subject_ids[:-1]

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

# Load a dataframe from one or multiple subjects
def load_dataframe(subject_ids):
    file_contents = []
    loaded_files = []
    for subject_id in subject_ids:
        print(subject_id)
        with zipfile.ZipFile("../public_dataset/" + subject_id+".zip") as file:
            contents = []
            for file_name in file.namelist():
                if "Accelerometer" in file_name:
                    df = pd.read_csv(file.open(file_name))
                    df['SubjectID'] = subject_id
                    df.columns = ["SysTime", "EventTime", "ActivityID", "X", "Y", "Z", "Rotation", "SubjectID"]
                    keep = ["X","Y","Z","SubjectID"]
                    df = df[keep]
                    df = df.astype(float)
                    content = df.values
                    loaded_files.extend(divide_windows(content, 1500))
    loaded = np.stack(loaded_files)
    return loaded

def split(subject_id, data, size):
    X_train, X_test = train_test_split(data, test_size=size, shuffle=True, random_state=20)
    return X_train, X_test

# Save data to a given file
def save_file(data, filename):
    loc = "split/" + subject_id + "/" + filename
    if not os.path.exists(os.path.dirname(loc)):
        try:
            os.makedirs(os.path.dirname(loc))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    file = open(loc, "w+")

    for window in data:
        np.savetxt(file, window, delimiter=",")
    file.close()

if __name__ == "__main__":
    subject_ids = find_subject_ids("../public_dataset/")
    i = 0
    for subject_id in subject_ids:
        if(subject_id == "data_description" or subject_id == ""):
            continue
        i += 1
        df = load_dataframe([subject_id])

        # Split dataframe into development and validation data
        df_dev, df_val = split(subject_id, df, 0.1)

        # Split development data into train and test
        df_train, df_test = train_test_split(df_dev, test_size=.2, shuffle=True, random_state=20)

        #Save the splits to a new directory
        save_file(df_val, "df_val.csv")
        save_file(df_train, "df_train.csv")
        save_file(df_test, "df_test.csv")
        print("Done "+ subject_id + "   ||  Number completed: " + str(i) + "    ||  Percentage: " + str(float(i)/100. * 100) + "%")


