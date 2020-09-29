import os
import torch
import pickle
import numpy as np


class MusicCorpus:
    def __init__(self, data, converter, sequence_len=0, cuda=True, batch_size=None, balance=False):
        self.data_size = int(data.size()[0])
        self.data_dim = int(data.size()[1])
        self.sequence_len = sequence_len
        self.data = data
        self.cuda = cuda
        if cuda:
            self.data = self.data.cuda()
        if self.cuda:
            self.tensor_float = torch.cuda.FloatTensor
            self.tensor_long = torch.cuda.LongTensor
        else:
            self.tensor_float = torch.FloatTensor
            self.tensor_long = torch.LongTensor
        self.converter = converter
        self.batch_size = batch_size
        self.tgt_idx = 3

    def get_sequence(self, index):
        # index is the first note index in the sequence
        assert index <= (
                self.data_size - self.sequence_len), "Received index {0}. Index must be smaller than size of data: {1}" \
            .format(index, (self.data_size - self.sequence_len))
        data = torch.empty(self.sequence_len, self.data.shape[1], dtype=self.data.dtype, device=self.data.device)
        data[:, :3] = self.data[index:(index + self.sequence_len), :3]
        data[:, 3:] = self.data[(index + 1):(index + self.sequence_len + 1), 3:]
        target_vector = self.data[(index + 1):(index + self.sequence_len + 1), :3]
        return data, target_vector

    def get_random_batch(self):
        # returns a batch with random sequences
        batch_input = self.tensor_long(self.sequence_len, self.batch_size, self.data_dim)
        batch_target = self.tensor_long(self.sequence_len, self.batch_size, self.tgt_idx)
        data_size = self.data_size
        for i in range(self.batch_size):
            random_index = np.random.randint(0, data_size - self.sequence_len)
            batch_input[:, i, :], batch_target[:, i, :] = self.get_sequence(random_index)
        return batch_input, batch_target

    def get_batch(self, index):
        batch_input = self.tensor_long(self.sequence_len, self.batch_size, self.data_dim)
        batch_target = self.tensor_long(self.sequence_len, self.batch_size, self.tgt_idx)
        for i in range(self.batch_size):
            batch_input[:, i, :], batch_target[:, i, :] = self.get_sequence(
                index * self.batch_size * self.sequence_len + i * self.sequence_len)
        return batch_input, batch_target

    def get_num_batches(self):
        return int(((self.data_size // self.sequence_len) - 1) // self.batch_size)

    def get_num_sequences(self):
        return int((self.data_size // self.sequence_len) - 1)

    def __len__(self):
        return self.data_size

    def data_dim(self):
        return self.data_dim

    def cuda(self):
        self.data = self.data.cuda()


def create_music_corpus(pkl_path, sequence_len=32, cuda=True, all=False, batch_size=None,
                        music_corpus_class=MusicCorpus):
    with open(os.path.join(pkl_path, 'converter_and_duration.pkl'), 'rb') as input_conv:
        converter = pickle.load(input_conv)

    with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as input_train:
        train_data = pickle.load(input_train)
    with open(os.path.join(pkl_path, 'val.pkl'), 'rb') as input_val:
        val_data = pickle.load(input_val)

    if all:
        train_data = np.concatenate((train_data, val_data))

    npdict_2_np = lambda data: np.concatenate([v for k, v in data.items()], axis=0)
    train_data = npdict_2_np(train_data)
    val_data = npdict_2_np(val_data)

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(val_data)

    train_corpus = music_corpus_class(train_data, converter, sequence_len=sequence_len, cuda=cuda,
                                      batch_size=batch_size)
    val_corpus = music_corpus_class(val_data, converter, sequence_len=sequence_len, cuda=cuda, batch_size=batch_size)

    return train_corpus, val_corpus, converter
