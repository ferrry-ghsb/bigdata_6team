import numpy as np

def make_sequence_dataset(
    train,
    valid,
    time_steps = 100,
    for_periods = 14
):
    ts_train_len = len(train)
    ts_valid_len = len(valid)
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len-1):
        X_train.append(train.iloc[i-time_steps:i])
        y_train.append(train.iloc[i:i+1, [0,2]].values.squeeze().tolist())
    

    X_valid = []
    y_valid = []
    for i in range(time_steps, ts_valid_len-1):
        X_valid.append(valid.iloc[i-time_steps:i])
        y_valid.append(valid.iloc[i:i+1, [0,2]].values.squeeze().tolist())
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)

    print(f'X_train.shape : {X_train.shape}, y_train.shape : {y_train.shape}')
    print(f'X_valid.shape : {X_valid.shape}, X_valid.shape : {X_valid.shape}')

    return X_train, X_valid, y_train, y_valid