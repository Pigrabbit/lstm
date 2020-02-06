from sklearn.preprocessing import StandardScaler
import numpy as np

TRAIN_SPLIT = 150
PAST_HISTORY = 5
FUTURE_TARGET = 0
STEP = 1

def scale(dataset):
    '''
    Scale the data set with Standard Scaler
    it gets mean and var from training set
    and fit entire dataset(trian + val set)
    INPUT: unscaled dataset
    OUTPUT: scaled dataset, scaler mean, scaler var
    '''
    scaler = StandardScaler()
    scaler.fit(dataset[:TRAIN_SPLIT])
    dataset = scaler.transform(dataset)
    return dataset, scaler.mean_, scaler.var_

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

def get_train_set(dataset):
    x_train, y_train = multivariate_data(
        dataset=dataset,
        target=dataset[:, 0],
        start_index=0,
        end_index=TRAIN_SPLIT,
        history_size=PAST_HISTORY,
        target_size=FUTURE_TARGET,
        step=STEP,
        single_step=True
        )
    return x_train, y_train

def get_val_set(dataset):
    x_val, y_val = multivariate_data(
        dataset=dataset,
        target=dataset[:, 0],
        start_index=TRAIN_SPLIT,
        end_index=None,
        history_size=PAST_HISTORY,
        target_size=FUTURE_TARGET,
        step=STEP,
        single_step=True
        )
    return x_val, y_val