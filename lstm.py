import json
import tensorflow as tf
import numpy as np
import data_reader as reader 
import plotter
import data_processor as processor
import model
import os
from os.path import dirname, join as pjoin

def save_rmse(rmse, configs, well, model):
    result_dir = pjoin(configs["data"]["data_path"], "result", well+ "_" + model + ".txt")

    with open(result_dir, "w") as f:
        f.write(f"rmse: {rmse}\n")
        json.dump(configs, f, sort_keys=True, indent=4, ensure_ascii=False)
        f.close()

if __name__ == "__main__":
    # Setup
    configs = json.load(open('config.json', 'r'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Load and Read dataset
    print("Setting up dataset reader")
    data_dir = pjoin(configs["data"]["data_path"], configs["data"]["data_filename"])
    data = reader.load_data(data_dir)
    data_dic = reader.read_data(data, configs)    
    
    # Choose a well and model to train and infrenece
    print("Choosing well and model...")

    for target_well in range(1, configs["data"]["num_wells"]+1):
        for target_model in range(1, 2):
        # for target_model in range(1, configs["data"]["num_models"]+1):
            target_well = str(target_well)
            target_model = str(target_model)
            result_dir = pjoin(configs["data"]["data_path"], "result", target_well + "_" + target_model + ".txt")
            if not os.path.exists(result_dir):
                # start process
                print(f"Current Well: {target_well}, Model: {target_model}")
                dataset = reader.choose_well_and_model(data_dic, target_well, target_model)
                
                # Preprocess the data according to LSTM architecture
                print("Preprocessing data according to LSTM design...")
                scaled_dataset, scaler_min, scaler_scale = processor.scale(dataset, configs)

                x_train, y_train = processor.get_train_set(scaled_dataset, configs)
                x_val, y_val = processor.get_val_set(scaled_dataset, configs)

                # Batch preprocessed dataset
                print("Batch dataset...")
                train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_data = train_data.cache().shuffle(configs["training"]["buffer_size"])
                train_data = train_data.batch(configs["training"]["batch_size"]).repeat()
                train_data = train_data.prefetch(1)

                val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                val_data = val_data.batch(configs["training"]["batch_size"]).repeat()
    
                # Define the LSTM model
                print("Define the LSTM model")
                lstm_model = model.define_lstm_model(x_train, configs["model"])
                lstm_model.summary()
    
                # Train and Fit the model
                print("Start training the model...")

                lstm_model, lstm_history = model.fit_model(lstm_model, train_data, val_data, configs)
                plotter.save_train_history(lstm_history, configs, target_well, target_model)
    
                # Inference
                print("Infernce getting started...")
                y_hat = model.inference(lstm_model, x_val)

                total_x = processor.get_total_x(scaled_dataset, configs)
                y_hat_total = model.inference(lstm_model, total_x)
    
                # Evaluate
                print("Evaluate the model")
                rmse = processor.get_rmse(y_val, y_hat[:, -1])
                print(f"Test RMSE: {rmse}")
                y_hat_inverse = processor.get_inverse_scaled(y_hat[:, -1], scaler_min[0], scaler_scale[0])
                y_val_inverse = processor.get_inverse_scaled(y_val, scaler_min[0], scaler_scale[0])
                y_train_inverse = processor.get_inverse_scaled(y_train, scaler_min[0], scaler_scale[0])

                y_hat_total_inverse = processor.get_inverse_scaled(y_hat_total[:, -1], scaler_min[0], scaler_scale[0])

                plotter.save_prediction(y_hat_inverse, y_val_inverse, y_train_inverse, configs, target_well, target_model)
                plotter.save_total_prediction(y_hat_total_inverse, y_val_inverse, y_train_inverse, configs, target_well, target_model)
                # Save results
                print("Saving the rmse result to txt file...")
                save_rmse(rmse, configs, target_well, target_model)

                tf.keras.backend.clear_session()
    