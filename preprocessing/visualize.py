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

if __name__ == "__main__":
    df = read_file("100669")
    print(df.shape[0])
    plot_dataframe(df)
