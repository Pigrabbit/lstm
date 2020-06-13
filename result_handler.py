from os.path import join as pjoin
import numpy as np

class ResultHandler():
    def __init__(self, true_val, pred_val, well_to_learn, true_model):
        self._true_val = true_val
        self._pred_val = pred_val
        self._well_to_learn = well_to_learn
        self._true_model = true_model

    def __get_mean_absolute_percentage_error(self):
        # remove zeros in order to calculate MAPE
        y_true = np.array([y for y in self._true_val if y != 0])
        y_pred = np.array([y for y in self._pred_val if y != 0])
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"MAPE: {mape}")
        return mape

    def save_mape_to_csv(self, result_filename):
        mape = self.__get_mean_absolute_percentage_error()
        result_dir = pjoin('result', result_filename)
        with open(result_dir, "a") as f:
            f.write(f"{self._well_to_learn},{self._true_model},{mape}\n")
            f.close()