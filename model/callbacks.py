from tensorflow import keras
import os
from utils.utils import get_run_logdir
import matplotlib.pyplot as plt
import numpy as np
import itertools

def callbacks_list(dir_name, model_name='model.h5', patience = 10):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(dir_name, model_name), monitor = 'loss')
    #early_stopping_cb = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True, monitor='loss')
    tensorboard_cb = keras.callbacks.TensorBoard(os.path.join(dir_name, "tensorboard"), histogram_freq=1)
    #scheduler = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = 5)

    return [checkpoint_cb, tensorboard_cb]
