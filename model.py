from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.layers import GaussianNoise, LSTM, Dropout, Dense

def define_lstm_model(x_train, configs):
    lstm_model = models.Sequential()
    lstm_model.add(GaussianNoise(
        configs["layers"]["gaussian_std"],
        input_shape=x_train.shape[-2:]
        ))
    lstm_model.add(LSTM(
        configs["layers"]["lstm1_units"],
        activation=configs["layers"]["activation"],
        return_sequences=True
        ))
    lstm_model.add(Dropout(configs["layers"]["dropout_rate"]))

    lstm_model.add(LSTM(
        configs["layers"]["lstm2_units"],
        activation=configs["layers"]["activation"],
        return_sequences=True
        ))
    lstm_model.add(Dropout(configs["layers"]["dropout_rate"]))

    lstm_model.add(Dense(configs["layers"]["dense_units"]))

    lstm_model.compile(
        optimizer=configs["optimizer"],
        loss=configs["loss"]
        )

    return lstm_model

def early_stopping(pateince):
    return callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1)

def fit_model(model, train_data, val_data, configs):
    steps_per_epoch = configs["data"]["train_split"] // configs["training"]["batch_size"]
    validation_steps = configs["data"]["validation_split"] // configs["training"]["batch_size"]
    history = model.fit(
        train_data,
        epochs=configs["training"]["epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        use_multiprocessing=True,
        workers=configs["training"]["workers"],
        callbacks=[early_stopping(configs["training"]["es_patience"])]
    )
    return model, history

def inference(trained_model, x_val):
    return trained_model.predict(x_val)