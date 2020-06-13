import random
import numpy as np
import tensorflow as tf


class BatchReader():
    __NUM_MODEL = 104
    __NUM_FEATURES_USED = 1
    __INPUT_SEQUENCE = 5
    __FUTURE_TARGET = 0
    __STEP = 1
    __train_model_list = []
    __val_model_list = []
    __test_model_list = []
    __train_seq_length = 0

    def __init__(self, scaled_data, end_indice, train_split, true_model, buffer_size, batch_size):
        self._scaled_data = scaled_data
        self._end_indice = end_indice
        self._train_split = train_split
        self._true_model = true_model
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._split_dataset(self._true_model)

    def _split_dataset(self, test_model_idx):
        model_pool = list(range(1, self.__NUM_MODEL + 1))
        self.__test_model_list.append(test_model_idx)
        model_pool.remove(test_model_idx)
        for i in range(self._train_split):
            chosen = random.choice(model_pool)
            self.__train_model_list.append(chosen)
            model_pool.remove(chosen)
        self.__val_model_list.extend(model_pool)

    def _multivariate_data(self, dataset, target, start_index, end_index, history_size,
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

    def _get_input_and_label(self, serialized_well, model_list, end_indice):
        dataset_x = np.empty(
            (0, self.__INPUT_SEQUENCE, self.__NUM_FEATURES_USED))
        dataset_y = np.empty((0))

        for model in model_list:
            start_index = end_indice[str(model)][0]
            end_index = end_indice[str(model)][1]

            range_multi_x, range_multi_y = self._multivariate_data(
                serialized_well[start_index:end_index][:,
                                                       :self.__NUM_FEATURES_USED],
                serialized_well[start_index:end_index][:, 0],
                start_index=0,
                end_index=end_index - start_index,
                history_size=self.__INPUT_SEQUENCE,
                target_size=self.__FUTURE_TARGET,
                step=self.__STEP,
                single_step=True
            )

            dataset_x = np.concatenate((dataset_x, range_multi_x))
            dataset_y = np.concatenate((dataset_y, range_multi_y))

        return dataset_x, dataset_y

    def get_seq_length(self):
        return self.__train_seq_length

    def get_train_batch(self):
        train_x, train_y = self._get_input_and_label(
            self._scaled_data, self.__train_model_list, self._end_indice)
        self.__train_seq_length = train_x.shape[0]

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_data = train_data.cache().shuffle(
            self._buffer_size).batch(self._batch_size).repeat()
        train_data = train_data.prefetch(1)

        return train_data

    def get_val_batch(self):
        val_x, val_y = self._get_input_and_label(
            self._scaled_data, self.__val_model_list, self._end_indice)
        val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_data = val_data.batch(self._batch_size).repeat()

        return val_data

    def get_test_input_and_label(self, test_data):
        total_timestep = test_data.shape[0]
        test_x, test_y = self._multivariate_data(
            test_data[:, :self.__NUM_FEATURES_USED],
            test_data[:, 0],
            start_index=0,
            end_index=total_timestep,
            history_size=self.__INPUT_SEQUENCE,
            target_size=self.__FUTURE_TARGET,
            step=self.__STEP,
            single_step=True
        )

        return test_x, test_y