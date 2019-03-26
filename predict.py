from keras.models import load_model

import sys
sys.path.append("../")
from fifc.fifc_opt import FifcOpt
import numpy as np


feat_dim = (4, 4, 8)
feat_flat_dim = np.prod(np.array(feat_dim))

import sys


def main():
    save_weight_file = sys.argv[-1]
    model = load_model(save_weight_file)

    fifc_opt = FifcOpt("../fifc/font_img/image/hiragino/",
                       "../fifc/img_char/image_save_dict/",
                       "../fifc/img_feature/weight/char_feature.hdf5")

    char = "あお"
    feat_list = []
    for c in char:
        feat = fifc_opt.char2feat(c)
        feat_reshape = feat.reshape(feat_flat_dim)
        feat_list.append(feat_reshape)
    feat_list = np.array([feat_list])

    for _ in range(10):
        feat_result = model.predict(feat_list)
        feat_result_last = feat_result[0, -1].reshape(1, 1, feat_flat_dim)
        feat_list = np.append(feat_list, feat_result_last, axis=1)

    for feat in feat_list[0]:
        feat = feat.reshape(feat_dim[0], feat_dim[1], feat_dim[2])
        char = fifc_opt.feat2char(feat)
        print(char)


if __name__ == "__main__":
    main()
