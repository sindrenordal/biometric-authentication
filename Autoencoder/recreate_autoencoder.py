import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense

'''
Information from: "Smartphone Continuous Authentication Using Deep Learning Autoencoders"
Window-size = 500
Input-dim = 500 * 3
Number of hidden units each layer = 1500
Number of layers = 5
'''

def create_model():
    encoding_dim = 1500

    input_layer = Input(shape=(encoding_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    encoder = Dense(int(encoding_dim/2), activation="tanh")(encoder)
    decoder = Dense(int(encoding_dim/2), activation="tanh")(encoder)
    decoder = Dense(int(encoding_dim), activation="tanh")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder


if __name__ == "__main__":
    model = create_model()
    print(model.summary())
