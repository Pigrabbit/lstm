from os.path import dirname, join as pjoin
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import data_reader as reader 
import plotter
import preprocessor
import model

DATA_PATH = './data'
DATA_FILE_NAME = 'Brugge_en_d.mat'
RESULT_FILE_NAME = 'rmse_brugge.csv'

BATCH_SIZE = 32
BUFFER_SIZE = 1000

def inference(trained_model, x_val):
    return trained_model.predict(x_val)

def inverse_std_scaler(scaled_val, mean, var):
    return scaled_val * np.sqrt(var) + mean

def get_rmse(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))

def save_rmse(well_num, model_num, rmse):
    result_dir = pjoin(DATA_PATH, RESULT_FILE_NAME)

    with open(result_dir, "a") as f:
        f.write(f"{well_num}, {model_num}, {rmse}")
        f.close()

if __name__ == "__main__":
    #########################################
    # Setup
    
    #########################################
    # Load and Read dataset
    print("Setting up dataset reader")
    data_dir = pjoin(DATA_PATH, DATA_FILE_NAME)
    data = reader.load_data(data_dir)
    data_dic = reader.read_data(data)
    # In case, you want to see WOPR of single well
    # print("Saving WOPR of single well...")
    # well_num = 9
    # plotter.save_plot_single_well(dataset_dic, well_num)
    
    ##########################################
    # Choose a well and model to train and infrenece
    print("Choosing well and model...")
    well_num = '9'
    model_num = '1'

    dataset = reader.choose_well_and_model(data_dic, well_num, model_num)
    ##########################################
    # Preprocess the data according to LSTM architecture
    print("Preprocessing data according to LSTM design...")
    scaled_dataset, scaler_mean, scaler_var = preprocessor.scale(dataset)

    x_train, y_train = preprocessor.get_train_set(scaled_dataset)
    x_val, y_val = preprocessor.get_val_set(scaled_dataset)
    ##########################################
    # Batch preprocessed dataset
    print("Batch dataset...")
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_data = train_data.prefetch(1)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()
    ##########################################
    # Define the LSTM model
    print("Define the LSTM model")
    lstm_model = model.define_lstm_model(x_train)
    lstm_model.summary()
    ##########################################
    # Train and Fit the model
    print("Start training the model...")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    lstm_model, lstm_history = model.fit_model(lstm_model, train_data, val_data, 150, BATCH_SIZE)

    title = f"Training and Validation loss of well_{well_num} model_{model_num}"
    fig_id=f"w{well_num}_m{model_num}_history"
    plotter.save_train_history(lstm_history, title, fig_id)
    ##########################################
    # Inference
    print("Infernce getting started...")
    y_hat = inference(lstm_model, x_val)

    ##########################################
    # Evaluate
    print("Evaluate the model")
    y_hat_inverse = inverse_std_scaler(y_hat[:, -1], scaler_mean[0], scaler_var[0])
    y_val_inverse = inverse_std_scaler(y_val, scaler_mean[0], scaler_var[0])
    y_train_inverse = inverse_std_scaler(y_train, scaler_mean[0], scaler_var[0])

    title = f"WOPR prediction of well_{well_num} model_{model_num}"
    fig_id=f"w{well_num}_m{model_num}_prediction"
    plotter.save_prediction(y_hat_inverse, y_val_inverse, y_train_inverse, title, fig_id)

    rmse = get_rmse(y_val_inverse, y_hat_inverse)
    ##########################################
    # Save results
    print("Saving the rmse result to csv file...")
    save_rmse(well_num, model_num, rmse)

    tf.keras.backend.clear_session()
    ##########################################
    