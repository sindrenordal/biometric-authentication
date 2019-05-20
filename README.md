# Biometric authentication of smartphone users using accelerometer data.

We present an LSTM-based model that achieves an AUC of 82% and an EER of 24.12% with a window length of three seconds.

The repo consists of three files:
1. split.py
2. train.py
3. test.py

Split.py performs preprocessing of the dataset. It splits the data for each user into three files. The files are saved as df_train.csv, df_test.csv, and df_val.csv.

Train.py performs the training of the model. It reads the files created by split.py, and uses the data to train one binary classifier for each user in the dataset. The classifiers are saved as subjectID.h5.

Test.py performs the validation of our test results. It uses the df_val.csv for each user to provide us with a validation result.

Dependecies:  
Pandas - https://pandas.pydata.org  
Numpy - https://www.numpy.org  
Scikit-learn - https://scikit-learn.org/stable/  
Tensorflow-gpu - https://www.tensorflow.org/  
Keras - https://keras.io  
Train.py and test.py is dependent of a GPU to run.  

User guide:  
1. Enable GPU support in Tensorflow. See their guide(https://www.tensorflow.org/install/gpu) for the setup for your specific software and hardware. You MUST have access to a GPU to run train.py and test.py.  
2. Navigate to the code repository in terminal.
3. python split.py
4. python train.py
5. python test.py
