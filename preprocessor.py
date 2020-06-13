import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, num_model, true_model):
        self._num_model = num_model
        self._true_model = true_model
        self._model_list = [str(model_idx) for model_idx in range(1, num_model + 1)]

    def remove_zero_wopr(self, target_well_data):
        for model_index in self._model_list:
            df = target_well_data[model_index]
            target_well_data[model_index] = df[df.WOPR != 0]

        return target_well_data

    def serialize_well_dataframe(self, target_well_data):
        serialized_well = np.empty((0, 4))
        end_indice = {}
        start = 0

        for model in self._model_list:         
            # skip true model
            if model == str(self._true_model):
                continue
            model_data = target_well_data[model].values
            serialized_well = np.concatenate((serialized_well, model_data))
            num_timesteps = model_data.shape[0]
            end_indice[model] = [start, start + num_timesteps]
            start += num_timesteps

        return serialized_well, end_indice

    def scale_serialzed_data(self, serialized_data):
        scaler = MinMaxScaler()
        scaler.fit(serialized_data)
        scaled_data = scaler.transform(serialized_data)

        return scaled_data, scaler
