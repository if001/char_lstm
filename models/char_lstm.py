"""
教師データとなる文章は改行区切りである必要はない。
BOSやEOSも必要ない。
"""

from keras.layers import Input, Dense, Conv2D, LSTM, ReLU, Dropout
from keras.models import Model
from keras import backend as K[]
import keras
from keras.callbacks import EarlyStopping, CSVLogger
import os
import sys
sys.path.append("../")
import numpy as np


feat_dim = 4 * 4 * 8


def model():
    sentence_dim = 10
    hidden_dim = 512
    input_sentence = Input(shape=(None, feat_dim))
    x = Dense(512, activation="sigmoid")(input_sentence)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    output_sentence = Dense(feat_dim, activation="relu")(x)
    model = Model(input_sentence, output_sentence)
    model.summary()
    return model


def load_sentence(file_path):
    with open(file_path) as f:
        char_list = list(f.read())

    while(' ' in char_list):
        char_list.remove(' ')
    while('\n' in char_list):
        char_list.remove('\n')
    return char_list


def gen_train_data(char_list, to_vec_func, sentence_len, save_file_name=""):
    train = []
    teach = []
    for i in range(len(char_list) - 1 - sentence_len):
        char_vec_list = []
        for char in char_list[i:i + sentence_len + 1]:
            char_vec = to_vec_func(char)
            char_vec = char_vec.reshape(feat_dim)
            char_vec_list.append(char_vec)
        train.append(char_vec_list[:sentence_len])
        teach.append(char_vec_list[1:sentence_len + 1])

    train = np.array(train)
    teach = np.array(teach)
    print(train.shape)
    print(teach.shape)
    if save_file_name != "":
        np.savez(save_file_name, train=train, teach=teach)
        print("save :", save_file_name)
    return train, teach


def load_train_data(save_file_name):
    print(save_file_name)
    load = np.load(save_file_name)
    print("load :", save_file_name)
    return load['train'], load['teach']


def main():
    sentence_len = 10
    save_train_data_file = "./data/train_data.npz"
    save_weight_file = "./weight/char_lstm.hdf5"

    data_file = sys.argv[-1]
    root, ext = os.path.splitext(data_file)

    if ext == ".txt":
        char_list = load_sentence(data_file)
        fifc_opt = FifcOpt("../fifc/font_img/image/hiragino/",
                           "../fifc/img_char/image_save_dict/",
                           "../fifc/img_feature/weight/char_feature.hdf5")
        train, teach = gen_train_data(
            char_list, fifc_opt.char2feat, 10, save_train_data_file)
    elif ext == ".npz":
        train, teach = load_train_data(data_file)
    else:
        raise ValueError("load file format error")

    lstm_model = model()
    loss = 'mean_squared_error'
    opt = 'adam'
    lstm_model.compile(optimizer=opt,
                       loss=loss,
                       metrics=['acc'])
    es_cb = EarlyStopping(
        monitor='val_loss', patience=1, verbose=1, mode='auto')
    csv_logger = CSVLogger("./training_log.csv")

    lstm_model.fit(train, teach,
                   batch_size=256,
                   epochs=10,
                   verbose=1,
                   validation_split=0.2,
                   callbacks=[es_cb, csv_logger]
                   )
    lstm_model.save(save_weight_file)


if __name__ == "__main__":
    main()
