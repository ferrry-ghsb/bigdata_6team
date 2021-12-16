

from tensorflow import keras


def Model():
    model = keras.models.Sequential([
            keras.layers.LSTM(100, input_shape=[None, 3]),
            
            keras.layers.Dense(2)
    ])

    return model