from gensim.models import word2vec
import numpy as np


class DataSet():
    def __init__(self, train, teach, test_train=None, test_teach=None):
        self.train = train
        self.teach = teach
        self.test_train = test_train
        self.test_teach = test_teach


class DataLoader():
    def __init__(self, w2v, target_file, save_train_data_file, is_load=False):
        self.window_size = 20
        self.char_feat_dim = w2v.feat_size
        if is_load:
            train, teach = self.__loader(save_train_data_file)
        else:
            train, teach = self.__saver(w2v.char2vec,
                                        target_file,
                                        save_train_data_file)

        self.__data_set = DataSet(train, teach)

    def get_data_set(self):
        return self.__data_set

    def __print_shape(self, train, teach):
        print("train :", train.shape)
        print("teach :", teach.shape)

    def __saver(self, w2v, target_file, save_train_data_file):
        char_list = self.__load_sentence(target_file)
        train, teach = self.__reshape_data(w2v, char_list)
        np.savez(save_train_data_file, train=train, teach=teach)
        print("save ", save_train_data_file)
        return train, teach

    def __loader(self, save_train_data_file):
        data = np.load(save_train_data_file)
        print("load ", save_train_data_file)
        train = data['train']
        teach = data['teach']
        self.__print_shape(train, teach)
        return train, teach

    def __load_sentence(self, target_file):
        with open(target_file) as f:
            char_list = list(f.read())
        char_list = np.array(char_list)
        char_list = char_list[char_list != " "]
        char_list = char_list[char_list != "\n"]
        char_list = list(char_list)
        return char_list

    def __reshape_data(self, to_vec_func, char_list):
        train = []
        teach = []
        for i in range(len(char_list) - 1 - self.window_size):
            char_vec_list = []
            for char in char_list[i:i + self.window_size + 1]:
                char_vec = to_vec_func(char)
                char_vec = char_vec.reshape(self.char_feat_dim)
                char_vec_list.append(char_vec)
            train.append(char_vec_list[:self.window_size])
            teach.append(char_vec_list[1:self.window_size + 1])
        train = np.array(train)
        teach = np.array(teach)
        self.__print_shape(train, teach)
        return train, teach


def main():
    pass


if __name__ == "__main__":
    main()
