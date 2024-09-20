import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


class Autoencoder:
    def __init__(self, num_features, verbose=True, mse_threshold=0.5,
                 archi="U30,U10,U5,U15,U48",
                 l2_value=0.0001 , learning_rate=1e-5,
                 epochs=100, batch_size=128, loss='mse'):
        self.model = Sequential()
        self.mse_threshold = mse_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        regularisation = l2(l2_value)
        layers = archi.split(',')
        input_ = Input((num_features,))
        previous = input_
        for l in layers:
            if l[0] == 'U':
                layer_value = int(l[1:])
                current = Dense(units=layer_value, activation='relu', use_bias=True, kernel_regularizer=regularisation,
                                kernel_initializer='uniform')(previous)
                current = BatchNormalization()(current)
                previous = current

        self.encoded = Dense(units=num_features, activation='linear')(previous)
        self.model = Model(input_, self.encoded)

        # Select loss function
        if loss == 'mse':
            loss_function = 'mean_squared_error'
        elif loss == 'mae':
            loss_function = 'mean_absolute_error'
        elif loss == 'binary_crossentropy':
            loss_function = 'binary_crossentropy'
        else:
            raise ValueError("Unsupported loss function. Choose 'mse', 'mae', or 'binary_crossentropy'.")

        self.model.compile(loss=loss_function, optimizer=Adam(learning_rate=learning_rate),
                           metrics=[self.accuracy])

        if verbose:
            self.model.summary()

    def accuracy(self, y_true, y_pred):
        mse = K.mean(K.square((y_true - y_pred)), axis=1)
        temp = K.ones_like(mse)
        return K.mean(K.equal(temp, K.cast(mse < self.mse_threshold, temp.dtype)))

    def train(self, x_train, x_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        return self.model.fit(x_train, x_train, shuffle=True, batch_size=self.batch_size, epochs=self.epochs,
                              validation_data=(x_val, x_val), callbacks=[early_stopping, reduce_lr])

    def get_encoded_features(self, x):
        encoder = Model(inputs=self.model.input, outputs=self.encoded)
        return encoder.predict(x)





