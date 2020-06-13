import numpy as np
import tensorflow as tf
import random
from absl import app
from absl import flags

from data_reader import DataReader
from preprocessor import Preprocessor
from model_builder import ModelBuilder
from history_logger import HistoryLogger
from plotter import Plotter
from batch_reader import BatchReader
from result_handler import ResultHandler

FLAGS = flags.FLAGS

flags.DEFINE_string("DATA_PATH", "data", "directory which has dataset")
flags.DEFINE_string("DATA_FILENAME", "Brugge_en_d.mat", "filename of dataset")
flags.DEFINE_string("RESULT_FILENAME", "mape.csv",
                    "filename of resulf csv file")
flags.DEFINE_integer("BATCH_SIZE", 32, "batch size for training")
flags.DEFINE_integer("BUFFER_SIZE", 1000, "buffer size for training")
flags.DEFINE_integer("EPOCHS", 5, "epoch for training")
flags.DEFINE_integer("TRUE_MODEL", 88, "select ground truth of certain well")
flags.DEFINE_integer("WELL_TO_LEARN", 9, "well index to train and inference")
flags.DEFINE_integer("OBSERVATION_DAY", 150,
                     "begin cascade inference from this date")
flags.DEFINE_integer("NUM_MODEL", 104, "number of models in each well")
flags.DEFINE_integer("TRAIN_SPLIT", 80, "train spilt")
flags.DEFINE_integer("EVALUATION_INTERVAL", 10, "evaluation interval")
flags.DEFINE_integer("INPUT_SEQUENCE", 5, "input sequence of LSTM")


def main(argv=None):
    random.seed(2)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # Load data and preprocess data
    print("Loading data...")
    data_reader = DataReader(FLAGS.DATA_PATH, FLAGS.DATA_FILENAME, FLAGS.NUM_MODEL)
    well_dic = data_reader.create_well_dictionary()

    print("Preprocessing data...")
    target_well = well_dic[str(FLAGS.WELL_TO_LEARN)]
    test_model_data = target_well[str(FLAGS.TRUE_MODEL)]

    preprocessor = Preprocessor(FLAGS.NUM_MODEL, FLAGS.TRUE_MODEL)
    well_data_zero_removed = preprocessor.remove_zero_wopr(target_well)
    serialized_data, end_indice = preprocessor.serialize_well_dataframe(
        well_data_zero_removed)
    scaled_data, scaler = preprocessor.scale_serialzed_data(serialized_data)

    # Split dataset and prepare batch
    batch_reader = BatchReader(scaled_data=scaled_data, end_indice=end_indice, train_split=FLAGS.TRAIN_SPLIT,
                               true_model=FLAGS.TRUE_MODEL, buffer_size=FLAGS.BUFFER_SIZE, batch_size=FLAGS.BATCH_SIZE)

    train_data = batch_reader.get_train_batch()
    val_data = batch_reader.get_val_batch()
    train_total_seq_length = batch_reader.get_seq_length()

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
        print('epochs : ' + str(epoch_idx+1))
        model.fit(
            train_data,
            epochs=1,
            steps_per_epoch=train_total_seq_length/FLAGS.BATCH_SIZE,
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
    test_x, test_y = batch_reader.get_test_input_and_label(test_data)

    seq_in = test_x[FLAGS.OBSERVATION_DAY -
                    FLAGS.BATCH_SIZE:FLAGS.OBSERVATION_DAY, :, :]
    seq_out = test_x[:FLAGS.INPUT_SEQUENCE, :1, :].flatten(
    ).tolist() + test_y[:FLAGS.OBSERVATION_DAY+1].tolist()

    pred_count = test_x.shape[0] - FLAGS.OBSERVATION_DAY

    # Do Inference from Observationday
    for i in range(1, pred_count):
        sample_in = seq_in
        pred_out = model.predict(sample_in)
        seq_out.append(pred_out[-1, :].item())
        seq_in = test_x[FLAGS.OBSERVATION_DAY -
                        FLAGS.BATCH_SIZE+i:FLAGS.OBSERVATION_DAY+i, :, :]

    model.reset_states()

    # Evaluate
    print("Start evaluating the model...")
    seq_out_array = np.asarray(seq_out)
    prediction_val = (seq_out_array - scaler.min_[0]) / scaler.scale_[0]
    true_val = test_model_data['WOPR'].to_numpy()

    # Plot prediction result
    print("Saving prediction result...")
    plotter.plot_prediction(total_timestep, true_val, prediction_val)

    # Calculate error and save into file
    print("Calculate MAPE and save it to result file...")
    result_handler = ResultHandler(true_val=true_val, pred_val=prediction_val,
                                   well_to_learn=FLAGS.WELL_TO_LEARN, true_model=FLAGS.TRUE_MODEL)
    result_handler.save_mape_to_csv(FLAGS.RESULT_FILENAME)

    # Clear Session
    tf.keras.backend.clear_session()
    print("Done")


if __name__ == "__main__":
    app.run(main)
