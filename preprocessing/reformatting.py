from os import listdir
import zipfile

# Returns array with all the id's if the subjects
def find_subject_ids(dataset_name):
    subject_ids = []
    for file_name in listdir(dataset_name):
        subject_ids.append(file_name.split(".")[0])
    return subject_ids[:-1]

if __name__ == "__main__":
    ids = find_subject_ids("public_dataset/")
    print(ids)
