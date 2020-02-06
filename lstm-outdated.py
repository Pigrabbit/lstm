import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

DATA_PATH = './data'
DATA_FILE_NAME = 'Brugge_en_d.mat'
RESULT_FILE_NAME = 'rmse_brugge.csv'

IMAGE_PATH = './images'

data_dir = pjoin(DATA_PATH, DATA_FILE_NAME)
mat_contents = sio.loadmat(data_dir)

data = mat_contents['en_d'][0, 0]
TRUE_MODEL_INDEX = 103
NUM_WELL = 20       # P1-20
NUM_MODEL = 104         # 1-104

INPUT_SEQUENCE = 5
OUTPUT_SEQUENCE = 1 
TRAIN_SPLIT = 150
INPUT_DIMENSION = 4     # num of input features
BATCH_SIZE = 24
BUFFER_SIZE = 1000
LSTM_NUM_UNITS = 25
LSTM_LOSS = 'mae'
LSTM_OPTIMIZER = 'adam'
EPOCHS = 50
EVALUATION_INTERVAL = 200

tf.debugging.set_log_device_placement(True)

#  'well_num' => dfs_dic
def to_dic_of_df(data, num_producer, num_model):
    well_dic = {}
    for well_index in range(num_producer):      # well, Producer P1-P20
        # 'model_num' => dataframe
        model_dic = {}
        well_key = 'P' + str(well_index+1)
        for model_index in range(num_model):    # model, model 1-104
            well_data = np.array([
                data['WOPR'][0,0][well_key][:,model_index],
                data['WBHP'][0,0][well_key][:,model_index],
                data['WWCT'][0,0][well_key][:,model_index],
                data['WWPR'][0,0][well_key][:,model_index]
              ])
            # convert np array to dataframe
            # | date | WOPR | WBHP | WWCT | WWPR |
            # |------|------|------|------|------|
            # |  0.0 |  0.0 |  0.0 |  0.0 |  0.0 |
            # | .... | .... | .... | .... | .... |  
            # |3648.0| .... | .... | .... | .... |  
            well_data = well_data.T
            df = pd.DataFrame(
                data=well_data,
                index=data['TIME'].flatten(),
                columns=['WOPR', 'WBHP', 'WWCT', 'WWPR']
            )
            df.index.name = 'date'
            model_dic[str(model_index+1)] = df

        well_dic[str(well_index+1)] = model_dic
    return well_dic
    
def plot_single_well(well_dic, well_num):
    ax = plt.gca()
    for i in range(104):
        df = well_dic[str(well_num)][str(i+1)]
        if i == TRUE_MODEL_INDEX:
            df.plot(y='WOPR', ax=ax, color='red', legend=False)
        else:
            df.plot(y='WOPR', ax=ax, color='gray', legend=False)
            
    plt.show()

def series_to_supervised(dataframe, n_in=1, n_out=1, dropnan=True):
    '''
    Frame a time series as a supervised learning dataset.
    Arguments:
        dataframe: Sequence of observations as a dataframe
        n_in: Number of lag observation as input, number of sequence
        n_out: Number of observations as output, output dimension
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    '''
    n_vars = len(dataframe.columns)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(dataframe.shift(i))
        names += [('var%d(t-%d)' %(j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, ... , t+n)
    for i in range(0, n_out):
        cols.append(dataframe.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_date_index(time_df, train_split):
    return time_df[train_split-1][0]


def scale_dataset(scaler, target_df, date_index):
    # Get a dataframe as parameter and minmax scale it
    # and return the result dataframe
    reframed = series_to_supervised(target_df, n_in=INPUT_SEQUENCE, n_out= OUTPUT_SEQUENCE)
    scaler = scaler.fit(reframed.loc[:date_index,:])
    reframed.loc[:, :] = scaler.transform(reframed.loc[:, :])
    return reframed


def split_train_test(reframed_df):
    reframed_np = reframed_df.to_numpy()
    train, test = reframed_np[:TRAIN_SPLIT, :], reframed_np[TRAIN_SPLIT:, :]
    # last 4 columns are data of cuurrent time step
    train_X, train_y = train[:, :-4], train[:, -4]
    test_X, test_y = test[:, :-4], test[:, -4]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], INPUT_SEQUENCE, INPUT_DIMENSION))
    test_X = test_X.reshape((test_X.shape[0], INPUT_SEQUENCE, INPUT_DIMENSION))
    return train_X, train_y, test_X, test_y


def save_train_history(history, title, fig_id, fig_extension='png', resolution=300):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    path = pjoin(IMAGE_PATH, fig_id + "." + fig_extension)  
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.clf()

def save_prediction(eval_array, time_df, scale, title, fig_id, fig_extension='png', resolution=300):
    df = pd.DataFrame(data=eval_array.T, columns=['prediction', 'true'])
    
    plt.figure()

    plt.plot(time_df[5:], df[['prediction']] * scale, linestyle='-', label='prediction')
    plt.plot(time_df[5:], df[['true']] * scale, linestyle='-',label = 'true')
    plt.title(title)
    plt.legend()

    path = pjoin(IMAGE_PATH, fig_id + "." + fig_extension)
    plt.savefig(path, format=fig_extension, dip=resolution)
    plt.clf()

def calculate_rmse(y_prediction, y_true, scaler, reframed):
    # invert scaling for forecast
    test = reframed.to_numpy()[TRAIN_SPLIT:, :]

    inv_y_hat = np.concatenate((y_prediction, test[:, 1:]), axis = 1)
    inv_y_hat = scaler.inverse_transform(inv_y_hat)
    inv_y_hat = inv_y_hat[:, 0]

    y_true = y_true.reshape((len(y_true), 1))
    inv_y = np.concatenate((y_true, test[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_y_hat))
    return rmse


well_dic = to_dic_of_df(
    data = data,
    num_producer=NUM_WELL,
    num_model=NUM_MODEL
)

scaler = MinMaxScaler()
date_index = get_date_index(data['TIME'], TRAIN_SPLIT)

# iterate over wells
#### iterate over models
for well_index in range(20, NUM_WELL+1):
    for model_index in range(1, NUM_MODEL+1): 
        target_df = well_dic[str(well_index)][str(model_index)]
        reframed = scale_dataset(scaler, target_df, date_index)
        train_X, train_y, test_X, test_y = split_train_test(reframed)

        train_data = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        val_data = val_data.batch(BATCH_SIZE).repeat()

        # define and fit the model using keras
        model = Sequential()
        model.add(LSTM(
            units = LSTM_NUM_UNITS, 
            input_shape=(INPUT_SEQUENCE, INPUT_DIMENSION)
        ))
        model.add(Dense(1))

        model.compile(loss = LSTM_LOSS, optimzer = LSTM_OPTIMIZER)
        history = model.fit(
            train_data,
            epochs=EPOCHS, 
            steps_per_epoch=EVALUATION_INTERVAL,
            validation_data=val_data,
            validation_steps=50
            )

        save_train_history(
            history=history, 
            title=f"Well {well_index} Model {model_index} training and validation loss",
            fig_id=f"w{well_index}_m{model_index}_history"
            ) 

        # Evaluate the model
        y_hat = model.predict(test_X)
        test_X = test_X.reshape(test_X.shape[0], INPUT_SEQUENCE * INPUT_DIMENSION)

        # scale factor of label
        scale = 1 / scaler.scale_[-4]

        evaluate_array = np.array([
            np.concatenate((train_y, y_hat), axis=None),
            np.concatenate((train_y, test_y), axis=None)
            ])

        time_df = data['TIME']
        

        save_prediction(
            eval_array=evaluate_array, 
            time_df=time_df,
            scale=scale,
            title=f"Well {well_index} Model {model_index} prediction",
            fig_id=f"w{well_index}_m{model_index}_prediction"
            )

        rmse = calculate_rmse(
            y_prediction=y_hat,
            y_true=test_y,
            scaler=scaler,
            reframed=reframed
        )

        print(f"RMSE: {rmse}")
        
        dir = pjoin(DATA_PATH, RESULT_FILE_NAME)
        f = open(dir, "a")
        f.write(f"{well_index}, {model_index}, {rmse}\n")
        f.close()

        tf.keras.backend.clear_session()
