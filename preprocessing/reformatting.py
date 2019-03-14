import os
import zipfile
import numpy as np
import pandas as pd

# Returns array with all the id's if the subjects
def find_subject_ids(dataset_name):
    subject_ids = []
    for file_name in os.listdir(dataset_name):
        subject_ids.append(file_name.split(".")[0])
    return subject_ids[:-1]

# Gathers all accelerometerdata for each user into a single csv
def reformat_one_subject(subject_id):
    file_list = []
    with zipfile.ZipFile("public_dataset/"+subject_id+".zip") as file:
       file_contents = []
       for file_name in file.namelist():
           if "Accelerometer" in file_name:
               with file.open(file_name) as accelerometer_file:
                   for line in accelerometer_file:
                       line_cleaned = (subject_id +","+ line.decode("utf-8").rstrip())
                       file_contents.append(line_cleaned.split(","))

    content = np.array(file_contents)
    columns_accelerometer = ["SubjectID", "SysTime", "EventTime", "ActivityID", "X", "Y", "Z", "Rotation"]
    df = pd.DataFrame(data = content, columns = columns_accelerometer)
    df.to_csv("Accelerometer_data/"+subject_id+".csv", index=False)
         
        
if __name__ == "__main__":
    ids = find_subject_ids("public_dataset/")
    for id in ids:
        print(id)
        reformat_one_subject(id)
