from tensorflow.keras import models, layers, callbacks

LSTM1_UNITS = 50
LSTM2_UNITS = 50
GAUSSIAN_STD = 0.01
DROPOUT_RATE = 0.2
OPTIMIZER = 'adam'
LOSS_METRIC = 'mean_squared_error'
EPOCHS = 200
VALIDATION_STEPS = 50

def define_lstm_model(x_train):
    lstm_model = models.Sequential()
    lstm_model.add(layers.GaussianNoise(GAUSSIAN_STD, input_shape=x_train.shape[-2:]))
    lstm_model.add(layers.LSTM(LSTM1_UNITS, activation='relu', return_sequences=True))
    lstm_model.add(layers.Dropout(DROPOUT_RATE))

    lstm_model.add(layers.LSTM(LSTM2_UNITS, activation='relu', return_sequences=True))
    lstm_model.add(layers.Dropout(DROPOUT_RATE))

    lstm_model.add(layers.Dense(1))

    lstm_model.compile(optimizer=OPTIMIZER, loss=LOSS_METRIC)
    return lstm_model

def early_stopping():
    return callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1)

def fit_model(model, train_data, val_data, train_split, batch_size):
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=train_split//batch_size,
        validation_data=val_data,
        validation_steps=VALIDATION_STEPS,
        use_multiprocessing=True,
        workers=8,
        callbacks=[early_stopping()]
    )
    return model, history