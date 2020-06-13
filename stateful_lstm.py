import numpy as np
import tensorflow as tf
import random

from absl import app
from absl import flags
from tensorflow import keras
from os.path import dirname, join as pjoin

from reader import Reader
from preprocessor import Preprocessor
from model_builder import ModelBuilder
from history_logger import HistoryLogger
from plotter import Plotter

FLAGS = flags.FLAGS

flags.DEFINE_string("DATA_PATH", "data", "directory which has dataset")
flags.DEFINE_string("DATA_FILENAME", "Brugge_en_d.mat", "filename of dataset")
flags.DEFINE_string("RESULT_FILENAME", "mape.csv", "filename of resulf csv file")
flags.DEFINE_integer("BATCH_SIZE", 32, "batch size for training")
flags.DEFINE_integer("BUFFER_SIZE", 1000, "buffer size for training")
flags.DEFINE_integer("EPOCHS", 5, "epoch for training")
flags.DEFINE_integer("TRUE_MODEL", 88, "select ground truth of certain well")
flags.DEFINE_integer("WELL_TO_LEARN", 9, "well index to train and inference")
flags.DEFINE_integer("OBSERVATION_DAY", 150, "begin cascade inference from this date")
flags.DEFINE_integer("NUM_MODEL", 104, "number of models in each well")
flags.DEFINE_integer("TRAIN_SPLIT", 80, "train spilt")
flags.DEFINE_integer("EVALUATION_INTERVAL", 10, "evaluation interval")
flags.DEFINE_integer("NUM_FEATURES", 4, "number of columns in dataset")
flags.DEFINE_integer("NUM_FEATURES_USED", 1, "number of features used for LSTM")
flags.DEFINE_integer("INPUT_SEQUENCE", 5, "input sequence of LSTM")
flags.DEFINE_integer("FUTURE_TARGET", 0 ,"future target value of multivariate data")
flags.DEFINE_integer("STEP", 1, "prediction step in LSTM")

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
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

def split_train(test_model):
    model_list = list(range(1, FLAGS.NUM_MODEL + 1))
    train = []
    test = [test_model]
    model_list.remove(test_model)
    for i in range(FLAGS.TRAIN_SPLIT):
        chosen = random.choice(model_list)
        train.append(chosen)
        model_list.remove(chosen)

    val = model_list
    return train, val, test

def get_dataset(serialized_well, model_list, end_indice):
    dataset_x = np.empty((0, FLAGS.INPUT_SEQUENCE, FLAGS.NUM_FEATURES_USED))
    dataset_y = np.empty((0))

    for model in model_list:
        start_index = end_indice[str(model)][0]
        end_index = end_indice[str(model)][1]
    
        range_multi_x, range_multi_y = multivariate_data(
            serialized_well[start_index:end_index][:, :FLAGS.NUM_FEATURES_USED],
            serialized_well[start_index:end_index][:, 0],
            start_index = 0,
            end_index = end_index - start_index,
            history_size = FLAGS.INPUT_SEQUENCE,
            target_size = FLAGS.FUTURE_TARGET,
            step = FLAGS.STEP,
            single_step = True
        )
        
        dataset_x = np.concatenate((dataset_x, range_multi_x))
        dataset_y = np.concatenate((dataset_y, range_multi_y))

    return dataset_x, dataset_y

def get_mean_absolute_percentage_error(y_true, y_pred):
    # remove zeros in order to calculate MAPE
    y_true = np.array([y for y in y_true if y != 0])
    y_pred = np.array([y for y in y_pred if y != 0])
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def main(argv=None):
    random.seed(2)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # Load data and preprocess data
    print("Loading data...")
    data_reader = Reader(FLAGS.DATA_PATH, FLAGS.DATA_FILENAME, FLAGS.NUM_MODEL)
    well_dic = data_reader.create_well_dictionary()

    print("Preprocessing data...")
    target_well = well_dic[str(FLAGS.WELL_TO_LEARN)]
    test_model_data = target_well[str(FLAGS.TRUE_MODEL)]

    preprocessor = Preprocessor(FLAGS.NUM_MODEL, FLAGS.TRUE_MODEL)

    well_data_zero_removed = preprocessor.remove_zero_wopr(target_well)
    serialized_data, end_indice = preprocessor.serialize_well_dataframe(well_data_zero_removed)
    scaled_data, scaler = preprocessor.scale_serialzed_data(serialized_data)

    # Split dataset and prepare batc
    train_model_list, val_model_list, test_model_list = split_train(
        test_model=FLAGS.TRUE_MODEL)

    train_x, train_y = get_dataset(scaled_data, train_model_list, end_indice)
    val_x, val_y = get_dataset(scaled_data, val_model_list, end_indice)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.cache().shuffle(FLAGS.BUFFER_SIZE).batch(FLAGS.BATCH_SIZE).repeat()
    train_data = train_data.prefetch(1)

    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_data = val_data.batch(FLAGS.BATCH_SIZE).repeat()

    # Define Model
    print("Defining model...")
    model_builder = ModelBuilder(FLAGS.BATCH_SIZE)
    model = model_builder.contruct_model()
    model.summary()

    # Set Training callbacks
    history_logger = HistoryLogger()
    
    # Train the model
    print("Begin training the model...")
    for epoch_idx in range(FLAGS.EPOCHS):
        print ('epochs : ' + str(epoch_idx+1))
        model.fit(
            train_data,
            epochs=1,
            steps_per_epoch=train_x.shape[0]/FLAGS.BATCH_SIZE,
            verbose=2,
            validation_data=val_data,
            validation_steps=100,
            use_multiprocessing=True,
            callbacks=[history_logger]
        )
        model.reset_states()

    # Save fig of loss history
    print("Saving loss history")
    plotter = Plotter(FLAGS.EPOCHS, FLAGS.WELL_TO_LEARN, FLAGS.TRUE_MODEL)
    plotter.plot_loss_history(history_logger.losses, history_logger.val_losses)

    # Inference (Cascade)
    print("Starting inference...")
    test_data = scaler.transform(test_model_data.values)
    total_timestep = test_data.shape[0]
    test_x, test_y = multivariate_data(
        test_data[:, :FLAGS.NUM_FEATURES_USED],
        test_data[:, 0],
        start_index = 0,
        end_index = total_timestep,
        history_size=FLAGS.INPUT_SEQUENCE,
        target_size=FLAGS.FUTURE_TARGET,
        step=FLAGS.STEP,
        single_step=True
    )

    seq_in = test_x[FLAGS.OBSERVATION_DAY-FLAGS.BATCH_SIZE:FLAGS.OBSERVATION_DAY,:,:]
    seq_out = test_x[:FLAGS.INPUT_SEQUENCE,:1,:].flatten().tolist() + test_y[:FLAGS.OBSERVATION_DAY+1].tolist()

    pred_count = test_x.shape[0] - FLAGS.OBSERVATION_DAY

    for i in range(1, pred_count):
        sample_in = seq_in
        pred_out = model.predict(sample_in)
        seq_out.append(pred_out[-1,:].item())
        seq_in = test_x[FLAGS.OBSERVATION_DAY-FLAGS.BATCH_SIZE+i:FLAGS.OBSERVATION_DAY+i,:,:]    

    model.reset_states()

    # Evaluate 
    print("Start to evaluate the model...")
    seq_out_array = np.asarray(seq_out)
    prediction_val = (seq_out_array - scaler.min_[0])/ scaler.scale_[0]
    true_val = test_model_data
    true_val = true_val['WOPR'].to_numpy()

    # Plot prediction result
    print("Saving prediction result...")
    plotter.plot_prediction(total_timestep, true_val, prediction_val)

    # Calculate error and save into file
    print("Calculate MAPE and save it to result file...")
    mape = get_mean_absolute_percentage_error(true_val, prediction_val)
    print(f"MAPE: {mape}")

    result_dir = pjoin('result', FLAGS.RESULT_FILENAME)
    with open(result_dir, "a") as f:
        f.write(f"{FLAGS.WELL_TO_LEARN},{FLAGS.TRUE_MODEL},{mape}\n")
        f.close()

    # Clear Session
    tf.keras.backend.clear_session()
    print("Done")

if __name__ == "__main__":
    app.run(main)