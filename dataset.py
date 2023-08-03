import numpy as np
import keras
import math


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, drugs, prots, Y, pro2vec, batch_size, max_prot_len, index2char, shuffle=False):
        'Initialization'
        self.max_prot_len = max_prot_len
        self.pro_emb = 20
        self.batch_size = batch_size
        self.pro2vec = pro2vec
        self.index2char = index2char
        self.drugs, self.prots, self.Y = drugs, prots, Y
        self.indexes = [i for i in range(len(self.drugs))]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(math.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if (len(indexes)) == self.batch_size:
            td = np.empty((self.batch_size, 200))
            tp = np.empty((self.batch_size, 2000, 20))
            y = np.empty((self.batch_size, 1))
        else:
            td = np.empty((len(indexes), 200))
            tp = np.empty((len(indexes), 2000, 20))
            y = np.empty((len(indexes), 1))
        b_index = 0
        for i in indexes:
            pro_vec = self.get_pro_vec(self.prots[i])
            td[b_index] = self.drugs[i, :]
            tp[b_index] = pro_vec
            y[b_index] = self.Y[i]
            b_index += 1

        X = [td, tp]
        return X, y

    def get_pro_vec(self, prot):
        pro_vec = np.empty((self.max_prot_len, self.pro_emb))

        for i, ch_index in enumerate(prot):
            if int(ch_index) == 0:
                vec = np.zeros(self.pro_emb)
            else:
                ch = self.index2char[int(ch_index)]
                vec = self.pro2vec[ch]
            pro_vec[i, :] = vec
        return pro_vec

    def on_epoch_end(self):
        pass
        'Updates indexes after each epoch'
