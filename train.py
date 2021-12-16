from tensorflow import keras
import tensorflow as tf
from utils.utils import get_run_logdir
import os
from model.callbacks import *
import numpy as np
import sklearn
import io
from model.model import Model
from utils.preprocessing import make_sequence_dataset
import pandas as pd

def train(model,
    X_train, y_train,
    X_valid = None, y_valid = None,
    epochs = 500,
    learning_rate = 0.001,
    optimizer = keras.optimizers.Adam(), 
    dir_name = None,
    model_name = "model.h5",
    batcch_size = 32,
    patience = 10):

    if dir_name == None:
        dir_name = get_run_logdir()
        os.mkdir(dir_name)

    optimizer.learning_rate = learning_rate
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer = optimizer)

    history = model.fit(X_train, y_train,  validation_data=(X_valid, y_valid), epochs=epochs, callbacks=callbacks_list(dir_name, model_name, patience))

    return history


if __name__=='__main__':
    train_ = pd.read_csv('./data/bea_train.csv').iloc[:, 1:]
    valid_ = pd.read_csv('./data/bea_test.csv').iloc[:, 1:]

    model = Model()

    X_train, X_valid, y_train, y_valid = make_sequence_dataset(train_, valid_)

    train(model, X_train, y_train, X_valid, y_valid)