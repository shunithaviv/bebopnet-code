import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jazz_rnn.utils.utils import LockedDropout, WeightDrop
from jazz_rnn.utils.music.vectorXmlConverter import input_2_groups

OFFSET_TO_5OCT = 72


class BaseRnnModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, hidden_size, num_layers,
                 pitch_sizes, duration_sizes, root_sizes,
                 offset_sizes, scale_sizes, chord_sizes,
                 tie_weights=True, normalize=False,
                 wdrop=0.5, dropouti=0.0, dropouth=0.3, dropoute=0.1):
        super(BaseRnnModel, self).__init__()

        self.drop = nn.Dropout(dropouth)
        self.lockdrop = LockedDropout()
        self.normalize = normalize

        self.encode_pitch = nn.Embedding(pitch_sizes[0], pitch_sizes[1], scale_grad_by_freq=True)
        self.encode_duration = nn.Embedding(duration_sizes[0], duration_sizes[1])
        self.encode_offset = nn.Embedding(offset_sizes[0], offset_sizes[1])

        lstm_input_size = self.set_encoding_layers(pitch_sizes, duration_sizes, root_sizes,
                                                   offset_sizes, scale_sizes, chord_sizes)

        self.rnns = nn.ModuleList([nn.LSTM(lstm_input_size if l == 0 else hidden_size
                                           , hidden_size, 1, dropout=0) for l in range(num_layers)])
        if wdrop != 0:
            self.rnns = nn.ModuleList([WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns])

        self.linear_pitch = nn.Linear(hidden_size, pitch_sizes[1])
        self.linear_duration = nn.Linear(hidden_size, duration_sizes[1])

        self.decode_pitch = nn.Linear(pitch_sizes[1], pitch_sizes[0])
        self.decode_octave = nn.Linear(hidden_size, 10)
        self.decode_duration = nn.Linear(duration_sizes[1], duration_sizes[0])

        # Optionally tie weights as in:
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decode_pitch.weight = self.encode_pitch.weight
            self.decode_duration.weight = self.encode_duration.weight

        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.wdrop = wdrop

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = pitch_sizes[0] + duration_sizes[0]

    @abstractmethod
    def set_encoding_layers(self, pitch_sizes, duration_sizes, root_sizes,
                            offset_sizes, scale_sizes, chord_sizes, ):
        pass

    @abstractmethod
    def forward_embed(self, bptt, batch_size, root, chord_pitches,
                      scale_pitches, pitch_emb, duration_emb,
                      offset_emb, chord_idx, rank):
        pass

    def forward_on_output(self, output):
        return output

    @staticmethod
    def get_chord_pitches_emb(batch_size, bptt, chord_pitches):
        try:
            chord_pitches = chord_pitches.nonzero()[:, 2].contiguous().view(bptt, batch_size, 4) + OFFSET_TO_5OCT
        except RuntimeError:
            # when there isn't a chord present in the XML, a special no-chord pitch is selected rather
            # than 4 pitches. The embed chord layer expects 4 pitches. this is a problem!
            # To handle this, in the ugly code below, the no-chord pitch is added 4 times.
            new_chord_pitches = []
            for e in chord_pitches.nonzero()[:, 2].data.cpu().numpy():
                if e != 12:
                    new_chord_pitches.append(e + OFFSET_TO_5OCT)
                else:
                    new_chord_pitches.extend([12] * 4)
            chord_pitches = chord_pitches.data.new(np.asarray(new_chord_pitches)).view(bptt, batch_size, 4)

        return chord_pitches

    def get_chord_pitch_emb(self, batch_size, bptt, chord_pitches):
        chord_pitches = self.get_chord_pitches_emb(batch_size, bptt, chord_pitches)
        chord_pitches_flat = chord_pitches.view(-1)
        chord_pitches_emb = self.encode_pitch(chord_pitches_flat).view(bptt, batch_size, -1)
        return chord_pitches_emb

    def forward_reward(self, inputs, hidden):
        bptt, batch_size, _ = inputs.size()
        _, _, hidden = self.forward(inputs, hidden)
        attention_key = torch.cat((self.embedded.view(bptt * batch_size, -1), self.output), dim=1)
        attention_key = attention_key.view(bptt, batch_size, -1).permute(1, 0, 2).contiguous()
        attention_value = self.output.view(bptt, batch_size, -1).permute(1, 0, 2).contiguous()

        attention_out, attention_probs = attention(query=self.reward_attention.weight,
                                                   key=attention_key, value=attention_value,
                                                   dropout=self.dropouta)
        reward_logits = self.reward_linear(attention_out.squeeze(1))
        return reward_logits, hidden

    def forward(self, inputs, hidden):

        convert_tuple_2_list_of_tuples = lambda y: list(zip(*tuple(map(lambda x: torch.chunk(x, 3), y))))
        hidden = convert_tuple_2_list_of_tuples(hidden)

        bptt = inputs.size()[0]
        batch_size = inputs.size()[1]

        pitch, duration, offset, root, scale_pitches, chord_pitches, chord_idx, rank, octave = \
            input_2_groups(inputs, bptt, batch_size)


        pitch_emb = self.encode_pitch(pitch)
        duration_emb = self.encode_duration(duration)
        offset_emb = self.encode_offset(offset)

        self.embedded = self.forward_embed(bptt, batch_size, root, chord_pitches,
                                           scale_pitches, pitch_emb, duration_emb,
                                           offset_emb, chord_idx, rank)

        emb = self.lockdrop(self.embedded, self.dropouti)

        raw_output = emb
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            if l != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropouth)

        output = output.view(output.size(0) * output.size(1), output.size(2))

        self.output = self.forward_on_output(output)

        output_pitch = self.drop(torch.sigmoid(self.linear_pitch(self.output)))
        output_duration = self.drop(torch.sigmoid(self.linear_duration(self.output)))

        if self.normalize:
            output_pitch = F.normalize(output_pitch, p=2, dim=1)
            output_duration = F.normalize(output_duration, p=2, dim=1)

        decoded_pitch = self.decode_pitch(output_pitch)
        decoded_duration = self.decode_duration(output_duration)

        convert_list_of_tuples_2_stacked_tensors = lambda y: \
            tuple(map(lambda x: torch.stack(x).squeeze(1), tuple(zip(*y))))
        hidden = convert_list_of_tuples_2_stacked_tensors(hidden)
        return decoded_pitch, decoded_duration, hidden

    def forward_rnn(self, inputs, hidden):

        convert_tuple_2_list_of_tuples = lambda y: list(zip(*tuple(map(lambda x: torch.chunk(x, 3), y))))
        hidden = convert_tuple_2_list_of_tuples(hidden)

        bptt = inputs.size()[0]
        batch_size = inputs.size()[1]

        pitch, duration, offset, root, scale_pitches, chord_pitches, chord_idx, rank, octave = \
            input_2_groups(inputs, bptt, batch_size)

        pitch_emb = self.encode_pitch(pitch)
        duration_emb = self.encode_duration(duration)
        offset_emb = self.encode_offset(offset)

        embedded = self.forward_embed(bptt, batch_size, root, chord_pitches,
                                      scale_pitches, pitch_emb, duration_emb,
                                      offset_emb, chord_idx, rank)

        emb = self.lockdrop(embedded, self.dropouti)

        raw_output = emb
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            if l != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropouth)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        output = self.forward_on_output(output)

        convert_list_of_tuples_2_stacked_tensors = lambda y: \
            tuple(map(lambda x: torch.stack(x).squeeze(1), tuple(zip(*y))))
        hidden = convert_list_of_tuples_2_stacked_tensors(hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).detach()
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())

    def norm_emb(self):
        self.encode_pitch.weight.data = F.normalize(self.encode_pitch.weight, p=2, dim=1)
        self.encode_duration.weight.data = F.normalize(self.encode_duration.weight, p=2, dim=1)


class ChordPitchesModel(BaseRnnModel):
    def __init__(self, *args, **kwargs):
        super(ChordPitchesModel, self).__init__(*args, **kwargs)

    def set_encoding_layers(self, pitch_sizes, duration_sizes, root_sizes,
                            offset_sizes, scale_sizes, chord_sizes):
        chord_pitches_size = 4 * pitch_sizes[1]
        lstm_input_size = pitch_sizes[1] + duration_sizes[1] + offset_sizes[1] + chord_pitches_size
        return lstm_input_size

    def forward_embed(self, bptt, batch_size, root, chord_pitches, scale_pitches,
                      pitch_emb, duration_emb, offset_emb, chord_idx, rank):
        chord_pitches_emb = self.get_chord_pitch_emb(batch_size, bptt, chord_pitches)
        embedded = torch.cat((pitch_emb, chord_pitches_emb, duration_emb, offset_emb), 2)
        return embedded


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


rnn_models_dict = {'chord_pitches': ChordPitchesModel}
