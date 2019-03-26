from gensim.models import word2vec


class W2vOpts():
    def __init__(self, train_file, weight_file, is_load=False):
        self.feat_size = 128
        if is_load:
            model = self.__loader(weight_file)
        else:
            model = self.__saver(train_file, weight_file)
        self.__model = model

    def char2vec(self, char):
        res = self.__model.wv[char]
        return res

    def vec2char(self, vec):
        res = self.__model.most_similar([vec], [], 1)[0][0]
        return res

    def __w2v_train(self, train_file):
        sentences = word2vec.Text8Corpus(train_file)
        model = word2vec.Word2Vec(
            sentences, size=self.feat_size, window=50, workers=4, min_count=1, hs=1)
        return model

    def __loader(self, weight_file):
        model = word2vec.Word2Vec.load(weight_file)
        print("load ", weight_file)
        return model

    def __saver(self, train_file, weight_file):
        model = self.__w2v_train(train_file)
        model.save(weight_file)
        print("save w2v:", weight_file)
        return model
