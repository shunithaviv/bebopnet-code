import os
from collections import Counter, OrderedDict
import pickle

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit


class RewardMusicCorpus:
    def __init__(self, data, converter, cuda=True, batch_size=None, balance=True, n_classes=2, seq_len=None):
        if cuda:
            self.tensor_float = torch.cuda.FloatTensor
            self.tensor_long = torch.cuda.LongTensor
        else:
            self.tensor_float = torch.FloatTensor
            self.tensor_long = torch.LongTensor
        self.n_classes = n_classes
        self.converter = converter
        self.batch_size = batch_size

        new_song_data_list = []
        for song_data in data:
            n_samples, n_features = song_data.shape
            new_song_data = song_data[:(seq_len * (n_samples // seq_len))].reshape(-1, seq_len, n_features)
            if n_samples % seq_len != 0:
                last_sequence = song_data[-seq_len:].reshape(1, seq_len, n_features)
                new_song_data = np.concatenate((new_song_data, last_sequence), axis=0)
            new_song_data_list.append(new_song_data)

        self.data = torch.from_numpy(np.concatenate(new_song_data_list, axis=0))
        self.sequence_labels = self.data[:, -1, -1].float()

        self.data = self.data.float()
        self.sequence_labels = self.sequence_labels - 50
        self.sequence_labels = self.sequence_labels / 50
        self.data[:, -1, -1] = self.sequence_labels

        self.n_sequences, self.seq_len, self.sample_size = self.data.shape
        self.n_features = self.sample_size - 1
        self.sequence_indices_pool = torch.arange(self.n_sequences)
        self.embedding = None
        if balance:
            self.balance_dataset()
            self.n_sequences = len(self.sequence_indices_pool)
            self.permute_order()

        if cuda:
            self.cuda()

    def permute_order(self):
        self.sequence_indices_pool = np.random.permutation(self.sequence_indices_pool)

    def balance_dataset(self):
        non_neutral_idxs = (self.sequence_labels - 1).nonzero()
        ros = RandomOverSampler()

        if self.n_classes == 2:
            sequence_labels = self.sequence_labels[non_neutral_idxs]
            idxs = non_neutral_idxs
        else:
            sequence_labels = self.sequence_labels
            idxs = torch.arange(len(self.sequence_labels)).unsqueeze(1)
        indices_resampled, labels_resampled = ros.fit_resample(idxs, sequence_labels.squeeze())
        self.sequence_indices_pool = indices_resampled

    def balancing_check(self):
        c = Counter()
        for i in range(10):
            self.balance_dataset()
            for ind in range(self.get_num_batches()):
                _, tar = self.get_batch(ind)
                c.update(tar.cpu().numpy().flatten().tolist())
            print('epoch', i)
            print(c)
            print('c0/c2={:.2f}'.format(c[0] / c[2]))
            print('c0/c1={:.2f}'.format(c[0] / c[1]))
            print('c1/c2={:.2f}'.format(c[1] / c[2]))
            c.clear()

        print(c)
        exit()

    def get_sequence(self, index):
        data = torch.empty(self.seq_len, self.n_features, dtype=self.data.dtype, device=self.data.device)
        data[:, :3] = self.data[index, :, :3].squeeze()
        data[:-1, 3:] = self.data[index, 1:, 3:-1]
        data[-1, 3:] = self.data[index, -1, 3:-1]
        target = self.data[index, :, -1].unsqueeze(-1)[-1]

        return data, target

    def get_batch(self, index):
        batch_input = self.tensor_long(self.seq_len, self.batch_size, self.n_features)
        batch_target = self.tensor_float(1, self.batch_size, 1)
        for i in range(self.batch_size):
            seq_ind = int(self.sequence_indices_pool[index * self.batch_size + i])
            batch_input[:, i, :], batch_target[:, i, :] = self.get_sequence(seq_ind)
        return batch_input, batch_target

    def get_num_batches(self):
        return int((self.n_sequences) // self.batch_size)

    def cuda(self):
        self.data = self.data.cuda()

    def __len__(self):
        return self.n_sequences


def smooth(y, box_pts=5):
    # box_pts has to be uneven number
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def quantize_label(data_dict, smooth_labels=False):
    for k, data in data_dict.items():
        quantized_label = data[:, -1]
        if smooth_labels:
            quantized_label = smooth(quantized_label, box_pts=9)
        quantized_label = quantized_label - 50
        quantized_label[np.abs(quantized_label) < 20] = 0
        quantized_label = np.sign(quantized_label) + 1
        data[:, -1] = quantized_label


def create_music_corpus(pkl_path, all=False, load_indices=False, val_ensemble=True):
    with open(os.path.join(pkl_path, 'converter_and_duration.pkl'), 'rb') as input_conv:
        converter = pickle.load(input_conv)

    with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as input_train:
        train_data = pickle.load(input_train)
    with open(os.path.join(pkl_path, 'val.pkl'), 'rb') as input_val:
        val_data = pickle.load(input_val)

    all_data = OrderedDict({**train_data, **val_data})
    all_data_list = [v for k, v in all_data.items()]

    data_indices_path = os.path.join(pkl_path, 'data_indices.pkl')

    ensemble_val_idxs = None
    if val_ensemble:
        if load_indices:
            with open(os.path.join(pkl_path, 'ensemble_idxs.pkl'), 'rb') as f:
                data_cv_idxs, ensemble_val_idxs = pickle.load(f)
        else:
            song_labels = np.zeros(len(all_data_list))
            for i, song in enumerate(all_data_list):
                seq_labels = np.zeros(song[:, -1].shape[0])
                seq_labels[song[:, -1] > 60] = 1
                seq_labels[song[:, -1] < 40] = -1
                song_labels[i] = stats.mode(seq_labels)[0][0]
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=17)
            data_cv_idxs, ensemble_val_idxs = next(sss.split(np.zeros(len(song_labels)), song_labels))
            with open(os.path.join(pkl_path, 'ensemble_idxs.pkl'), 'wb') as f:
                pickle.dump((data_cv_idxs, ensemble_val_idxs), f)
    else:
        data_cv_idxs = np.arange(len(all_data_list))

    if all:
        idxs_kf = [(np.array(list(range(len(all_data_list)))), np.array([]))]
    elif load_indices:
        with open(data_indices_path, 'rb') as f:
            idxs_kf = pickle.load(f)
    else:
        kf = KFold(n_splits=5, shuffle=False, random_state=0)
        idxs_kf = kf.split(data_cv_idxs)
        idxs_kf = list(idxs_kf)

        idxs_kf = [(np.sort(data_cv_idxs[x]), np.sort(data_cv_idxs[y])) for x, y in idxs_kf]

        with open(data_indices_path, 'wb') as f:
            pickle.dump(idxs_kf, f)

    return all_data_list, idxs_kf, converter, ensemble_val_idxs
