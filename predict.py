import os
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def period_prediction(model, X_valid, on_promotion, for_periods):
    predict = []
    X_pred = X_valid.copy()
    for i in range(for_periods):
        pred = model.predict(np.expand_dims(X_pred, axis = 0))
        
        predict += pred.tolist()
        X_pred = np.append(X_pred[1:], np.expand_dims(np.insert(pred, 1, on_promotion[i]), axis = 0), axis = 0)
    
    return np.array(predict)


def predict(
    model, X = 'sales_promotion_oil.csv', promotion = 'promotion.csv', for_periods = 14, path = './', name = 'predict'):
    
    if type(model) == str:
        model = keras.models.load_model(model)
    X_valid = pd.read_csv(X).values[:, 1:]
    promotion = pd.read_csv(promotion).values[:, 1]
    time_steps = X_valid.shape[0]
    predict = period_prediction(model, X_valid, promotion, for_periods)

    plt.figure(figsize= (15, 4))
    
    # period
    plt.plot(np.arange(len(X_valid)), X_valid[:, 0])

    # True
    #plt.plot(np.arange(time_steps, time_steps + for_periods), valid.iloc[i+time_steps:time_steps + for_periods+i, 0].values)

    # predict
    plt.plot(np.arange(time_steps, time_steps + for_periods) , predict[:, 0])
    plt.plot(np.arange(time_steps, time_steps + for_periods) , promotion, alpha = 0.4)

    plt.vlines(time_steps, 0, 1, color = 'gray', linestyle = '--')
    #plt.vlines(time_steps + 7, 0, 1, color = 'red', linestyle = '--')
    plt.ylim(X_valid[:, 0].min() - 0.1,X_valid[:, 0].max() + 0.2)
    plt.title(f'time_steps : {time_steps}, periods : {for_periods}')
    plt.grid()
    plt.legend(['period', 'Predict', 'Promotion'])
    plt.savefig(os.path.join(path, name))
    plt.show()