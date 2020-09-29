import os
import copy
import pickle

import numpy as np
import torch
from jazz_rnn.A_data_prep.gather_data_from_xml import REST_SYMBOL, EOS_SYMBOL

TRANSPOSE_PITCHES = np.arange(-5, 7)


def transpose_data_torch(p, data):
    shape = data.shape
    tranposed_data = copy.deepcopy(data)

    tranposed_data = tranposed_data.view(-1, tranposed_data.shape[-1])
    not_rest_or_eos_mask = (tranposed_data[:, 0] != REST_SYMBOL) & (tranposed_data[:, 0] != EOS_SYMBOL)
    tranposed_data[not_rest_or_eos_mask, 0] = tranposed_data[not_rest_or_eos_mask, 0] + p
    tranposed_data[:, 3] = (tranposed_data[:, 3] + p) % 12
    tranposed_data[:, 4:16] = torch.roll(tranposed_data[:, 4:16], shifts=int(p), dims=1)
    tranposed_data[:, 17:29] = torch.roll(tranposed_data[:, 17:29], shifts=int(p), dims=1)
    tranposed_data = tranposed_data.view(*shape)
    return tranposed_data


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.data_size = data.shape[1]
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1, self.data_size).permute(1, 0, 2).contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)
        _, bsz, d_data = self.data.shape

        beg_idx = max(0, i - self.ext_len)
        end_idx = i + seq_len

        data = torch.empty(seq_len, bsz, d_data, dtype=self.data.dtype, device=self.data.device)
        data[:, :, :3] = self.data[beg_idx:end_idx, :, :3]
        data[:, :, 3:] = self.data[i + 1:i + 1 + seq_len, :, 3:]

        # The note before EOS should not take the eos chord, rather should maintain its own.
        # Likewise, the EOS note, should maintain his zero vector for a chord
        eos_mask = self.data[beg_idx:end_idx + 1, :, 0] == EOS_SYMBOL
        eos_mask[:-1] += eos_mask[1:]
        eos_mask = eos_mask[:-1]
        eos_mask_expanded = eos_mask.unsqueeze(-1).expand(-1, -1, d_data - 3)
        data[:, :, 3:][eos_mask_expanded] = self.data[beg_idx:end_idx][:, :, 3:][eos_mask_expanded]
        target = self.data[i + 1:i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class JazzCorpus:
    def __init__(self, pkl_path, transpose=True):
        with open(os.path.join(pkl_path, 'converter_and_duration.pkl'), 'rb') as input_conv:
            self.converter = pickle.load(input_conv)

        with open(os.path.join(pkl_path, 'train.pkl'), 'rb') as input_train:
            train_data = pickle.load(input_train)
        with open(os.path.join(pkl_path, 'val.pkl'), 'rb') as input_val:
            val_data = pickle.load(input_val)

        self.transpose = transpose

        def npdict_2_np(data):
            return np.concatenate([v for k, v in data.items()], axis=0)

        train_data = npdict_2_np(train_data)
        val_data = npdict_2_np(val_data)

        self.train_data = torch.from_numpy(train_data)
        self.val_data = torch.from_numpy(val_data)

        train_data_transposed = []
        val_data_transposed = []
        for i in range(-5, 7):
            train_data_transposed.append(transpose_data_torch(i, self.train_data))
            val_data_transposed.append(transpose_data_torch(i, self.val_data))
        self.train_data = torch.cat(train_data_transposed)
        self.val_data = torch.cat(val_data_transposed)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            data_iter = LMOrderedIterator(self.train_data, *args, **kwargs)
        elif split == 'val':
            data_iter = LMOrderedIterator(self.val_data, *args, **kwargs)
        else:
            raise ValueError('split must be train|val')

        return data_iter
