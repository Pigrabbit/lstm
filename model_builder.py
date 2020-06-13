from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class ModelBuilder:
    _LSTM_UNITS = 128
    _INPUT_SEQUENCE = 5 
    _STEP = 1
    _DROPOUT_RATE = 0.25
    _DENSE_UNITS = 1

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def contruct_model(self):
        model = Sequential()
        model.add(LSTM(
            self._LSTM_UNITS, 
            batch_input_shape=(self._batch_size, self._INPUT_SEQUENCE, self._STEP), 
            activation='relu', 
            stateful=True
            ))
        model.add(Dropout(self._DROPOUT_RATE))
        model.add(Dense(self._DENSE_UNITS))

        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])

        return model