import matplotlib.pyplot as plt
from os.path import join as pjoin

class Plotter:
    __FIG_EXTENSION = "png"
    __IMG_DIR = "img"

    def __init__(self, epochs, well_to_learn, true_model):
        self._epochs = epochs
        self._well_to_learn = well_to_learn
        self._true_model = true_model

    def plot_loss_history(self, train_loss, val_loss):
        fig = plt.figure()

        plt.plot(range(1, self._epochs+1), train_loss, label='train')
        plt.plot(range(1, self._epochs+1), val_loss, label='validation')

        plt.ylabel('loss(Mean Squared Error)')
        plt.xlabel('epoch')
        plt.title(f"Train and validation losses, well:{self._well_to_learn}")
        plt.legend(loc='best')

        img_path = pjoin(self.__IMG_DIR, f"well{self._well_to_learn}_losses.{self.__FIG_EXTENSION}")
        fig.savefig(img_path, format=self.__FIG_EXTENSION, dpi=300)
        fig.clf()

    def plot_prediction(self, total_timestep, true_val, prediction_val):
        x_axis = range(total_timestep)

        fig = plt.figure()
        plt.plot(x_axis, true_val, label='ground truth')
        plt.plot(x_axis, prediction_val, label='prediction')

        plt.ylabel('WOPR')
        plt.xlabel('time')
        plt.title(f"WOPR forecasting, well: {self._well_to_learn}  true model: {self._true_model}")
        plt.legend(loc='best')

        img_path = pjoin(self.__IMG_DIR, f"well{self._well_to_learn}_model{self._true_model}_prediction.{self.__FIG_EXTENSION}")
        fig.savefig(img_path, format=self.__FIG_EXTENSION, dpi=300)
        fig.clf()