from w2v_opts import W2vOpts
from data_loader import DataLoader
from net_mng import NetMng
from models.model_set import ModelSet
import numpy as np
from keras.optimizers import SGD, RMSprop


def preprocessing(is_load=False):
    w2v_opts = W2vOpts(
        "./data/files_all_rnp-remove-space-split_space.txt",
        "./weight/test.model",
        is_load=is_load)

    data_loader = DataLoader(
        w2v_opts,
        # "./data/files_all_rnp.txt",
        # "./data/files_all_rnp.npz",
        # "./data/files_all_rnp_10000.txt",
        # "./data/files_all_rnp_10000.npz",
        "./data/test.txt",
        "./data/test.npz",
        is_load=is_load)
    if is_load is False:
        print("train end")
        exit(0)

    return w2v_opts, data_loader


def main():
    _, data_loader = preprocessing(is_load=True)

    data_set = data_loader.get_data_set()

    model_set = ModelSet()

    net_mng = NetMng(data_set, model_set.simple_lstm(), is_load=False)
    # net_mng = NetMng(data_set, model_set.test_lstm(), is_load=False)
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    net_mng.set_commpiler('mean_squared_error', opt)
    net_mng.train()

    x = np.array([data_set.train[0]])
    print(x)
    r = net_mng.predict(x)
    print(r)


if __name__ == "__main__":
    main()
