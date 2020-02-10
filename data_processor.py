import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

def scale(dataset, configs):
    '''
    Scale the data set with Standard Scaler
    it gets mean and var from training set
    and fit entire dataset(trian + val set)
    INPUT: unscaled dataset
    OUTPUT: scaled dataset, scaler mean, scaler var
    '''
    train_split = configs["data"]["train_split"]
    scaler = MinMaxScaler()
    scaler.fit(dataset[:train_split])
    dataset = scaler.transform(dataset)
    return dataset, scaler.min_, scaler.scale_

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      data.append(dataset[indices])

      if single_step:
        labels.append(target[i+target_size])
      else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def get_train_set(dataset, configs):
    x_train, y_train = multivariate_data(
        dataset=dataset,
        target=dataset[:, 0],
        start_index=0,
        end_index=configs["data"]["train_split"],
        history_size=configs["data"]["input_sequence"],
        target_size=configs["data"]["output_sequence"],
        step=configs["data"]["step"],
        single_step=True
        )
    return x_train, y_train

def get_val_set(dataset, configs):
    x_val, y_val = multivariate_data(
        dataset=dataset,
        target=dataset[:, 0],
        start_index=configs["data"]["train_split"],
        end_index=None,
        history_size=configs["data"]["input_sequence"],
        target_size=configs["data"]["output_sequence"],
        step=configs["data"]["step"],
        single_step=True
        )
    return x_val, y_val

def get_inverse_scaled(scaled_val, scaler_min, scaler_scale):
    return (scaled_val - scaler_min) / scaler_scale

def get_rmse(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))
