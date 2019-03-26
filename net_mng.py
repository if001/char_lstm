import os
from keras.callbacks import EarlyStopping, CSVLogger


class NetMng():
    def __init__(self, data_set, model, is_load=False):
        output_dir_base = os.path.join("./log/", model.name)
        os.makedirs(output_dir_base, exist_ok=True)
        self.__data_set = data_set
        self.__model = model

        self.__save_model_img = os.path.join(
            output_dir_base, "model.png")
        self.__train_log = os.path.join(output_dir_base, "train_log.csv")
        self.__weight_file = os.path.join(output_dir_base, "model.hdf5")
        if is_load:
            self.__model = self.__load_model

    def set_commpiler(self, loss, opt):
        # loss = 'mean_squared_error'
        # opt = 'adam'
        self.__model.compile(optimizer=opt,
                             loss=loss,
                             metrics=['acc'])

    def __callbacks(self):
        es_cb = EarlyStopping(
            monitor='val_loss', patience=1, verbose=1, mode='auto')
        csv_logger = CSVLogger(self.__train_log)
        return [es_cb, csv_logger]

    def train(self, epochs=2000, batch_size=256):
        self.__model.fit(self.__data_set.train,
                         self.__data_set.teach,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_split=0.2,
                         callbacks=self.__callbacks()
                         )
        self.__save_model()
        self.__plot_model()
        print("-- end --")

    def predict(self, data):
        return self.__model.predict(data)

    def __plot_model(self):
        from keras.utils import plot_model
        plot_model(self.__model,
                   to_file=self.__save_model_img, show_shapes=True)

    def __save_model(self):
        self.__model.save(self.__weight_file)
        print("save " + self.__weight_file)

    def __load_model(self):
        from keras.models import load_model
        model = load_model(self.__weight_file)
        print("load " + self.__weight_file)

        return model
