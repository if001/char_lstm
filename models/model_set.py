from keras.layers import Input, Dense, Conv2D, LSTM, ReLU, Dropout, BatchNormalization, Concatenate

from keras.layers.wrappers import Bidirectional as BD
from keras import activations
from keras.models import Model
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, CSVLogger

from keras.initializers import RandomUniform


import os
import sys
sys.path.append("../")
import numpy as np


class ModelSet():
    def __init__(self):
        self.input_shape = 128

    def test_lstm(self):
        sentence_dim = 10
        hidden_dim = 512
        input_sentence = Input(shape=(None, self.input_shape))
        x = LSTM(hidden_dim, return_sequences=True)(input_sentence)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        # x = Dense(hidden_dim, activation="tanh")(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        output_sentence = Dense(self.input_shape, activation="linear")(x)

        model = Model(input_sentence, output_sentence, name="simple_lstm")
        model.summary()
        return model

    def __lstm_block(self, x, unit):
        x = LSTM(unit, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        return x

    def __lstm_bd_block(self, x, unit):
        x = BD(LSTM(unit, return_sequences=True))(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        return x

    def __my_tanh(self, x):
        return 5 * K.tanh(x)

    def __my_tanh2(self, x):
        return (1 / 2) * K.tanh(x)

    def __cube_x(self, x):
        return x * x * x

    def __my_init(self):
        return RandomUniform(minval=-5, maxval=5, seed=None)

    def __route1(self, inp):
        x = Dense(256)(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)
        x = Dense(
            1024,
            activation=self.__my_tanh)(x)
        return x

    def __route2(self, inp):
        x = Dense(256)(inp)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.4)(x)
        x = Dense(
            1024,
            activation=self.__my_tanh2,
            kernel_initializer=self.__my_init())(x)
        return x

    def simple_lstm(self):
        hidden_dim = 64
        input_sentence = Input(shape=(None, self.input_shape))
        x = BatchNormalization()(input_sentence)
        x = LSTM(hidden_dim, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = LSTM(hidden_dim, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        # output_sentence = LSTM(self.input_shape,
        #                        return_sequences=True,
        #                        activation=self.__my_tanh)(x)
        x = Concatenate()([input_sentence, x])

        # x = Dense(
        #     256,
        #     activation='tanh')(x)
        # x = Dense(
        #     512,
        #     activation='tanh')(x)
        # x = Dense(
        #     1024,
        #     activation=self.__my_tanh,
        #     kernel_initializer=self.__my_init())(x)
        # x = Dropout(0.4)(x)
        r1 = self.__route1(x)
        r2 = self.__route2(x)
        x = Concatenate()([r1, r2])

        output_sentence = Dense(self.input_shape, activation="linear")(x)
        model = Model(input_sentence, output_sentence, name="simple_lstm")
        model.summary()
        return model

    def bd_lstm(self):
        hidden_dim = 64
        input_sentence = Input(shape=(None, self.input_shape))
        block = self.__lstm_bd_block(input_sentence, hidden_dim)
        x = self.__lstm_bd_block(block, hidden_dim)
        x = Concatenate()([block, x])
        output_sentence = LSTM(self.input_shape, return_sequences=True)(x)
        # x = Concatenate()([input_sentence, x])
        # x = Dense(hidden_dim * 2, activation="tanh")(x)
        # x = Dropout(0.4)(x)
        # x = Dense(hidden_dim * 2, activation="tanh")(x)
        # x = Dropout(0.4)(x)
        # output_sentence = Dense(self.input_shape, activation="tanh")(x)

        model = Model(input_sentence, output_sentence, name="simple_lstm")
        model.summary()
        return model
