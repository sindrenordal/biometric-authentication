import matplotlib.pyplot as plt
import os
import pandas as pd


def read_file(name):
    data_location = "../Accelerometer_data/"+name+".csv"
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), data_location))

    columns = ['X','Y','Z']
    df = df[columns]
    return df

def plot_dataframe(datafr):
    x = range(1,1513628)
    plt.plot(x, 'X', data = datafr, color='r')
    plt.plot(x, 'Y', data = datafr, color='g')
    plt.plot(x, 'Z', data = datafr, color='b')
    plt.show()

def plot_windows_results():
    auc = [0.8075, 0.8196, 0.8140, 0.7584]
    eer = [0.2571, 0.2412, 0.2509, 0.2928]
    window_sizes = [100, 300, 500, 1500]
    plt.plot(window_sizes, auc, color='r', label="AUC")
    plt.plot(window_sizes, eer, color='b', label="EER")
    plt.ylabel("Percentage")
    plt.xlabel("Window size")
    plt.legend()
    plt.grid(True)
    plt.show()

def show_dataframe():
    df = read_file("100669")
    print(df.shape[0])
    plot_dataframe(df)

plot_windows_results()
