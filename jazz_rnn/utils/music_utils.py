import copy
from fractions import Fraction

import music21 as m21
import numpy as np
import torch
import glob
import os
import pickle

from jazz_rnn.utils.music.vectorXmlConverter import create_note, tie_idx_2_value, tie_2_value
from jazz_rnn.utilspy.meters import AverageMeter

PITCH_IDX_IN_NOTE, DURATION_IDX_IN_NOTE, TIE_IDX_IN_NOTE, \
MEASURE_IDX_IN_NOTE, LOG_PROB_IDX_IN_NOTE = 0, 1, 2, 3, 4


def notes_to_stream(notes, stream, chords, head_len, remove_head=False, head_early_start=False):
    m = m21.stream.Measure()
    if remove_head:
        m.append(stream.flat.getElementsByClass(m21.tempo.MetronomeMark)[0])
        stream = m21.stream.Stream()
    else:
        stream = copy.deepcopy(stream)
    if head_early_start:
        measure_idx = -1
    else:
        measure_idx = 0
    # Insert chords to measure:
    measure_dur = 0
    note_num = 0
    for n in notes:
        note_num += 1
        if n[DURATION_IDX_IN_NOTE] == 0:
            continue
        m.append(
            create_note(n[PITCH_IDX_IN_NOTE], n[DURATION_IDX_IN_NOTE], tie_idx_2_value(n[TIE_IDX_IN_NOTE])))
        measure_dur += n[DURATION_IDX_IN_NOTE]
        if measure_dur >= 4 or note_num == notes.shape[0]:
            measure_dur = 0
            # Insert chords to measure:
            m.insert(0, copy.deepcopy(chords[measure_idx % len(chords)][0]))
            m.insert(2, copy.deepcopy(chords[measure_idx % len(chords)][1]))
            m.number = measure_idx + head_len + 1
            if not (remove_head and measure_idx == -1):
                stream.append(m)
            measure_idx += 1
            m = m21.stream.Measure()
    return stream


def notes_to_swing_notes(notes):
    print()
    notes = notes[notes[:, 1] != 0]
    durations = notes[:, 1]
    offsets = (np.cumsum(durations) - durations) % 4
    ind = 0
    while ind < len(durations) - 1:
        if offsets[ind] % 1 == 0:
            if durations[ind] == 0.5 and durations[ind + 1] == 0.5:
                num = np.random.randint(-2, 4)
                noise = Fraction(num / 48)
                durations[ind] = Fraction(14, 24) - noise
                durations[ind + 1] = Fraction(10, 24) + noise
            no_tie = notes[ind - 1, 2] == 0 and notes[ind, 2] == 0
            if ind > 0 and durations[ind] > Fraction(1, 3) and durations[ind - 1] > Fraction(1, 3) and no_tie:
                delay = Fraction(np.random.randint(-3, 1) / 64)
                if delay > 0:
                    durations[ind] = durations[ind] - delay
                    delay_note = copy.deepcopy(notes[ind])
                    delay_note[0] = notes[ind - 1, 0]
                    delay_note[1] = delay
                    delay_note[2] = tie_2_value['stop']
                    notes[ind - 1, 2] = tie_2_value['start']
                    np.insert(notes, ind, delay_note, axis=0)
                    np.insert(durations, ind, np.array([delay]))
                    np.insert(offsets, ind, offsets[ind])
                    ind += 1
                elif delay < 0:
                    delay = 0 - delay
                    durations[ind - 1] = durations[ind - 1] - delay
                    delay_note = copy.deepcopy(notes[ind - 1])
                    delay_note[0] = notes[ind, 0]
                    delay_note[1] = delay
                    delay_note[2] = tie_2_value['start']
                    notes[ind, 2] = tie_2_value['stop']
                    notes = np.concatenate((notes[:ind], delay_note[np.newaxis, :], notes[ind:]))
                    durations = np.concatenate((durations[:ind], np.array([delay]), durations[ind:]))
                    offsets = np.concatenate((offsets[:ind], np.array([offsets[ind]]), offsets[ind:]))
                    ind += 1
        ind = ind + 1
    notes[:, 1] = durations
    return notes


def get_topk_batch_indices_from_notes(notes, beam_width):
    number_of_notes_in_measure = np.count_nonzero(notes[:, :, LOG_PROB_IDX_IN_NOTE], axis=0)[np.newaxis, :]
    number_of_notes_in_measure = number_of_notes_in_measure.repeat(notes.shape[0], 0)
    measure_log_likelihood = (notes[:, :, LOG_PROB_IDX_IN_NOTE] / number_of_notes_in_measure).sum(axis=0)
    top_likelihood = measure_log_likelihood.max()
    sorted_indices = np.flip(measure_log_likelihood.argsort(), axis=0).copy()
    notes_set = set()
    topk_indices = []
    for ind in sorted_indices:
        set_size = len(notes_set)
        notes_set.add(tuple(notes[:, ind, :2].flatten()))
        if len(notes_set) == set_size + 1:
            topk_indices.append(ind)
        if len(topk_indices) == beam_width:
            break
    if len(topk_indices) < beam_width:
        for ind in sorted_indices:
            if ind not in topk_indices:
                topk_indices.append(ind)
                if len(topk_indices) == beam_width:
                    break

    return np.array(topk_indices), top_likelihood


class ScoreInference:
    def __init__(self, model_path, converter, beam_width, threshold, batch_size, ensemble=False):
        self.beam_width = beam_width
        self.mean_score_meters = [AverageMeter() for _ in range(batch_size)]
        self.ensemble = ensemble
        self.threshold = threshold
        self.converter = converter
        reward_converter_path = '/'.join(model_path.split('/')[:-1]) + '/converter_and_duration.pkl'
        with open(reward_converter_path, 'rb') as input_file:
            self.reward_converter = pickle.load(input_file)
        self.reward_supported_durs = list(self.reward_converter.bidict.keys())
        self.reward_durations = self.converter.dur_2_ind_vec(self.reward_supported_durs)
        self.reward_unsupported_durs = [i for i in range(len(self.converter.bidict)) if i not in self.reward_durations]
        if ensemble:
            self.model = []
            model_dir = '/'.join(model_path.split('/')[:-1])
            model_list = sorted(glob.glob(os.path.join(model_dir, "*f?.pt")), key=lambda x: int(x.split('/')[-1][-4]))
            for model_path in model_list:
                with open(model_path, 'rb') as f:
                    model = torch.load(f)
                self.model.append(model)
        else:
            with open(model_path, 'rb') as f:
                model = torch.load(f)
            self.model = model
        self.top_score = None

    def update(self, notes, update_mask):
        # change converters for score
        notes_to_update = copy.deepcopy(notes)
        notes_to_update[:, :, 1] = torch.Tensor(
            self.reward_converter.dur_2_ind_vec(self.converter.ind_2_dur_vec(notes[:, :, 1].view(-1)))).reshape(
            notes[:, :, 1].shape)
        scores = self.get_scores(notes_to_update)
        for ind, score in enumerate(scores):
            if update_mask[ind] == 1:
                score = score.item()
                # UNCOMMENT TO PUNISH LONG DURATIONS
                # if self.converter.ind_2_dur(notes[-1, ind, 1].item()) > 4:
                #     score = -100000
                self.mean_score_meters[ind].update(score)

    def get_scores(self, notes):
        scores = self.get_ensemble_score(notes)
        scores[torch.abs(scores) < self.threshold] = 0
        scores = torch.sign(scores)
        return scores

    def get_topk_batch_indices_from_notes(self, notes):
        measure_scores = np.array([m.avg for m in self.mean_score_meters])
        top_score = measure_scores.max()
        self.top_score = top_score
        sorted_indices = np.flip(measure_scores.argsort(), axis=0).copy()
        notes_set = set()
        topk_indices = []
        for ind in sorted_indices:
            set_size = len(notes_set)
            notes_set.add(tuple(notes[:, ind, :2].flatten()))
            if len(notes_set) == set_size + 1:
                topk_indices.append(ind)
            if len(topk_indices) == self.beam_width:
                break
        if len(topk_indices) < self.beam_width:
            for ind in sorted_indices:
                if ind not in topk_indices:
                    topk_indices.append(ind)
                    if len(topk_indices) == self.beam_width:
                        break

        new_meters = []
        while len(new_meters) != len(self.mean_score_meters):
            for k in topk_indices:
                new_meters.append(copy.deepcopy(self.mean_score_meters[k]))
        self.mean_score_meters = new_meters

        return np.array(topk_indices), top_score

    def get_ensemble_score(self, notes):
        scores_models = []
        for model in self.model:
            h = model.init_hidden(batch_size=notes.shape[1])
            scores, _ = model.forward_reward(notes, h)
            scores = torch.tanh(scores)
            scores_models.append(scores.squeeze())
        return torch.mean(torch.stack(scores_models), 0)


class HarmonyScoreInference:
    def __init__(self, converter, beam_width, beam_depth, batch_size):
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.converter = converter
        self.mean_score_meters = [AverageMeter() for _ in range(batch_size)]
        self.top_score = None

    def update(self, notes, update_mask):
        scores = self.get_sequence_score(notes)
        note_pitches = notes[1:, :, 0].squeeze()
        for ind, score in enumerate(scores):
            if update_mask[ind] == 1 and note_pitches[ind].item() != 128:
                self.mean_score_meters[ind].update(score.item())

    def get_sequence_score(self, notes):
        bptt, batch_size, _ = notes.shape

        notes_1_octave = notes[1:, :, 0] % 12

        durations = notes[1:, :, 1]
        durations_flat = durations.view(-1)
        durations_m21 = self.converter.ind_2_dur_vec(durations_flat)
        durations_m21 = durations_m21.reshape(1, batch_size)
        durations_float = torch.as_tensor(durations_m21.astype(np.float32)).cuda()
        chord_notes = torch.nonzero(notes.view(-1, 31)[:, 17:29])[:, 1].reshape(bptt, batch_size, 4)[:1]
        note_in_scale = torch.eq(notes_1_octave.unsqueeze(-1), chord_notes).sum(dim=-1)
        scores = (note_in_scale.float() / durations_float).squeeze()
        return scores

    def get_topk_batch_indices_from_notes(self, notes):
        measure_scores = np.array([m.avg for m in self.mean_score_meters])
        top_score = measure_scores.max()
        self.top_score = top_score
        sorted_indices = np.flip(measure_scores.argsort(), axis=0).copy()
        notes_set = set()
        topk_indices = []
        for ind in sorted_indices:
            set_size = len(notes_set)
            notes_set.add(tuple(notes[:, ind, :2].flatten()))
            if len(notes_set) == set_size + 1:
                topk_indices.append(ind)
            if len(topk_indices) == self.beam_width:
                break
        if len(topk_indices) < self.beam_width:
            for ind in sorted_indices:
                if ind not in topk_indices:
                    topk_indices.append(ind)
                    if len(topk_indices) == self.beam_width:
                        break

        new_meters = []
        while len(new_meters) != len(self.mean_score_meters):
            for k in topk_indices:
                new_meters.append(copy.deepcopy(self.mean_score_meters[k]))
        self.mean_score_meters = new_meters

        return np.array(topk_indices), top_score
