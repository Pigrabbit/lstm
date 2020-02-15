import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def save_plot_single_well(data_dic, configs, well, fig_extension='png', resolution=300):
    title = f"WOPR_well_{well}"
    image_path = configs["image"]["image_path"]
    image_dir = os.path.join(image_path, title + "." + fig_extension)
    if os.path.exists(image_dir):
        print("Plot already has been drawn and saved...")
    else:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(title)
        ax.set_xlabel("date")
        ax.set_ylabel("WOPR")
        true_model_num = 104
        for model_index in range(configs["data"]["num_models"]):
            df = data_dic[str(well)][str(model_index + 1)]
            if ((model_index + 1) == true_model_num):
                df.plot(y='WOPR', use_index=True, ax=ax, color='red', legend=False)
            else:
                df.plot(y='WOPR', use_index=True, ax=ax, color='gray', legend=False)

        fig.savefig(image_dir, format=fig_extension, dpi=resolution)
        fig.clf()
    
def save_train_history(history, configs, well, model, fig_extension='png', resolution=300):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    title = f"Training and Validation loss of well_{well} model_{model}"
    fig_id=f"w{well}_m{model}_history"
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(epochs, loss, 'b', label='Training loss')
    ax.plot(epochs, val_loss, 'r', label='Validation loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.legend()
    
    image_path = configs["image"]["image_path"]
    image_dir = os.path.join(image_path, 'history', fig_id + "." + fig_extension)
    fig.savefig(image_dir, format=fig_extension, dpi=resolution)
    fig.clf()

def save_prediction(y_predict, y_true, y_train, configs, well, model, fig_extension='png', resolution=300):
    y_train = y_train.reshape((-1, 1))
    y_predict = np.concatenate((y_train, y_predict), axis=None)
    y_true = np.concatenate((y_train, y_true.reshape((-1, 1))), axis=None)

    title = f"WOPR prediction of well_{well} model_{model}"
    fig_id=f"w{well}_m{model}_prediction"
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(y_predict, linestyle='-', label='prediction')
    ax.plot(y_true, linestyle='-', label='true')
    ax.set_xlabel('date')
    ax.set_ylabel('WOPR')
    ax.set_title(title)
    ax.legend()

    image_path = configs["image"]["image_path"]
    image_dir = os.path.join(image_path, 'prediction', fig_id + "." + fig_extension)
    fig.savefig(image_dir, format=fig_extension, dpi=resolution)
    fig.clf()

def save_total_prediction(y_predict, y_true, y_train, configs, well, model, fig_extension='png', resolution=300):
    y_train = y_train.reshape((-1, 1))
    # y_predict = np.concatenate((y_train, y_predict), axis=None)
    y_true = np.concatenate((y_train, y_true.reshape((-1, 1))), axis=None)

    title = f"WOPR prediction of well_{well} model_{model}"
    fig_id=f"w{well}_m{model}_total_prediction"
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(y_predict, linestyle='-', label='prediction')
    ax.plot(y_true, linestyle='-', label='true')
    ax.set_xlabel('date')
    ax.set_ylabel('WOPR')
    ax.set_title(title)
    ax.legend()

    image_path = configs["image"]["image_path"]
    image_dir = os.path.join(image_path, 'prediction', fig_id + "." + fig_extension)
    fig.savefig(image_dir, format=fig_extension, dpi=resolution)
    fig.clf()