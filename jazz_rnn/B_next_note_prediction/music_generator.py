from functools import lru_cache
from fractions import Fraction

from tqdm import tqdm
import numpy as np
import music21 as m21
import torch
import torch.nn.functional as F
import copy

from jazz_rnn.A_data_prep.gather_data_from_xml import extract_vectors, extract_chords_from_xml
from jazz_rnn.B_next_note_prediction.generation_utils import RES_LENGTH, early_start_song_dict
from jazz_rnn.utils.music.vectorXmlConverter import chord_2_vec, NOTE_VECTOR_SIZE, chord_2_vec_on_tensor, tie_2_value
from jazz_rnn.utilspy.meters import AverageMeter

PITCH_IDX_IN_NOTE, DURATION_IDX_IN_NOTE, TIE_IDX_IN_NOTE, \
MEASURE_IDX_IN_NOTE, LOG_PROB_IDX_IN_NOTE = 0, 1, 2, 3, 4

from jazz_rnn.utils.music_utils import get_topk_batch_indices_from_notes, ScoreInference, HarmonyScoreInference
from jazz_rnn.B_next_note_prediction.model import ChordPitchesModel

class MusicGenerator:
    """
    Generates jazz improvisations.
    Can generate solos in bulk, using beam search
    Args:
    -----
    model (nn.Module): nn to generate music
    converter (bidict): converter durations to ints
    beam_width (int): number of parallel measures to consider
    stochastic_search (bool): whether to choose the max possibility or to sample using softmax
    temperature (float): temperature for sampling new notes
    """

    def __init__(self, model, converter, batch_size,
                 beam_width, beam_depth, beam_search, non_stochastic_search, top_p,
                 temperature,
                 score_model='', threshold=None,
                 ensemble=False,
                 song='', no_head=False):

        self.filename = None
        self.song = song
        self.model = model
        self.model.normalize = False
        try:
            self.seq_len = model.max_klen
        except AttributeError:
            print('assuming bptt=16')
            self.seq_len = 16

        self.batch_size = batch_size
        self.beam_search = beam_search
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.device = next(model.parameters()).device
        self.converter = converter
        self.non_stochstic_search = non_stochastic_search
        self.top_p = top_p
        self.temperature = temperature

        self.chords = []
        self.initial_hidden = None

        self.reset_history()
        self.stream = m21.stream.Stream()
        self.head_len = 0
        self.get_non_tuplets_idxs_()
        self.initial_last_note_net = None
        self.notes_net = None

        self.pitch_var_meter = AverageMeter()
        self.dur_var_meter = AverageMeter()

        if score_model == 'harmony':
            self.seq_len = 2
            self.score_inference = HarmonyScoreInference(converter, beam_width, beam_depth, batch_size)
            self.score_model = 'harmony'
        elif score_model != '':
            self.score_inference = ScoreInference(score_model, converter, beam_width, threshold, batch_size,
                                                  ensemble=ensemble)
            self.score_model = 'reward'
        else:
            self.score_inference = None
            self.score_model = None

        self.no_head = no_head
        self.early_start = (not no_head) and (self.song in early_start_song_dict)

        self.notes_attention = None
        self.attentions = None

    @lru_cache(maxsize=1)
    def get_non_tuplets_idxs_(self):
        tuplets_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if (Fraction(dur_m21).denominator / 3) % 1 != 0:
                tuplets_list.append(dur_net)
        return torch.tensor(tuplets_list, device=self.device)

    @lru_cache(maxsize=1)
    def get_tuplets_idxs_(self):
        tuplets_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if (Fraction(dur_m21).denominator / 3) % 1 == 0:
                tuplets_list.append(dur_net)
        return torch.tensor(tuplets_list, device=self.device)

    @lru_cache(maxsize=1)
    def get_non_small_durs_idxs_(self):
        dur_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if (Fraction(dur_m21).denominator / 4) % 1 != 0:
                dur_list.append(dur_net)
        return torch.tensor(dur_list, device=self.device)

    @lru_cache(maxsize=1)
    def get_small_durs_idxs_(self):
        dur_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if (Fraction(dur_m21).denominator / 4) % 1 == 0:
                dur_list.append(dur_net)
        return torch.tensor(dur_list, device=self.device)

    def get_larger_durs_than_idxs_(self, x):
        dur_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if dur_m21 > x:
                dur_list.append(dur_net)
        return torch.tensor(dur_list, device=self.device)

    def get_non_x_dur_idxs_(self, x):
        dur_list = []
        for dur_m21, dur_net in self.converter.bidict.items():
            if dur_m21 != x:
                dur_list.append(dur_net)
        if len(dur_list) == len(self.converter.bidict.items()):
            return torch.tensor(self.converter.dur_2_ind(4), device=self.device)
        return torch.tensor(dur_list, device=self.device)

    def init_stream(self, xml_file):
        # read chords from xml
        self.chords = extract_chords_from_xml(xml_file)
        self.stream = m21.converter.parse(xml_file).parts[0]
        self.stream.autoSort = False
        self.insert_starting_notes(xml_file)
        self.head_len = len(self.chords)  # head_length can be defined in args for sequential generation

    def insert_starting_notes(self, xml_file):
        """insert notes to generator for the generator to continue"""
        data = extract_vectors(song=xml_file, ri=False, song_labels_dict={}, converter=self.converter)
        data = np.array(data)
        data = torch.as_tensor(data, device=self.device)
        data = data.unsqueeze(1).expand(data.shape[0], self.batch_size, data.shape[-1])

        # last note is <EOS>, so we don't want to to be pushed into the network.
        # otherwise it would ignore the entire input sequence
        if (data[-1, :, 0] == 129).all():
            data = data[:-1]

        self.notes_net = data[-self.seq_len:]
        self.notes_attention = data

        with torch.no_grad():
            if isinstance(self.model, ChordPitchesModel):
                _, _, self.initial_hidden = self.model(data, self.initial_hidden)
            else:
                _, _, self.initial_hidden, self.attentions = self.model._forward(data, [], self.initial_hidden)
        first_chord = self.chords[0][0]
        root, scale_pitches, chord_pitches, chord_idx = chord_2_vec(first_chord)
        self.initial_last_note_net = data.new(
            (list(data[-1, 0, :3]) + [root] + scale_pitches + chord_pitches + [chord_idx])).expand(1, self.batch_size,
                                                                                                   data.shape[-1])

    def fraction_2_float(self, ndarray):
        return np.vectorize(float)(ndarray)

    def object_to_tensor(self, ndarray, type, dtype):
        if ndarray.size != 0:
            ndarray = np.vectorize(type)(ndarray)
            return torch.as_tensor(ndarray, dtype=dtype, device=self.device)
        else:
            return torch.tensor([], dtype=dtype, device=self.device)

    def generate_measures(self, n_measures):
        """
        generates n_measures of music.
        will continue to generate measures on top of chords cyclically
        n_measures (int): number of measures to generate
        chords [[chords]]: chords to generate notes upon
        returns:
        -------
        m21.stream with notes and chords
        """
        # update generation starting point:
        notes = []
        residuals_m21 = np.ndarray([self.batch_size, RES_LENGTH], dtype=Fraction)

        # fill initial hidden and residuals_m21 (first measure)
        residuals_m21[:, :] = 0
        total_offset = np.zeros([self.batch_size], dtype=Fraction)
        last_note_net = self.initial_last_note_net
        hidden = self.initial_hidden

        with torch.no_grad():

            if self.song in early_start_song_dict and not self.no_head:
                short_xml = early_start_song_dict[self.song]
                self.stream = m21.converter.parse(short_xml).parts[0]
                self.stream.autoSort = False
                self.insert_starting_notes(short_xml)
                self.init_stream(short_xml)
                measure_idxs = [self.head_len - 1] + list(range(n_measures))
            else:
                measure_idxs = list(range(n_measures))

            for measure_idx in tqdm(measure_idxs):

                last_note_net, total_offset, hidden, residuals_m21, notes, top_score = self.generate_one_measure(
                    measure_idx,
                    n_measures, notes,
                    residuals_m21,
                    last_note_net,
                    total_offset,
                    hidden)

                n = np.array(notes)

                if self.beam_search == 'measure':
                    if measure_idx % self.beam_depth == 0:
                        if self.score_inference is not None:
                            topk_batch_indices, top_score = self.score_inference.get_topk_batch_indices_from_notes(
                                self.notes_net)
                            self.notes_net = self.general_repeat(self.notes_net, topk_batch_indices, dim=1)
                        else:
                            topk_batch_indices, top_score = get_topk_batch_indices_from_notes(n, self.beam_width)

                        # select k notes and repeat batch_size/beam_width times:
                        notes = list(self.general_repeat(n, topk_batch_indices, dim=1))
                        residuals_m21 = self.general_repeat(residuals_m21, topk_batch_indices, dim=0)
                        last_note_net = self.general_repeat(last_note_net, topk_batch_indices, 1)
                        total_offset = self.general_repeat(total_offset, topk_batch_indices, 0)
                        hidden = self.general_repeat(hidden, topk_batch_indices, 1)
                else:
                    if self.beam_search != 'note':
                        number_of_notes_in_measure = np.count_nonzero(n[:, :, LOG_PROB_IDX_IN_NOTE], axis=0)[
                                                     np.newaxis, :]
                        number_of_notes_in_measure = number_of_notes_in_measure.repeat(n.shape[0], 0)
                        measure_log_likelihood = (n[:, :, LOG_PROB_IDX_IN_NOTE] / number_of_notes_in_measure).sum(
                            axis=0)
                        top_score = measure_log_likelihood[0]

        if self.score_inference is not None:
            top_score = self.score_inference.top_score
        return np.array(notes), top_score

    def general_repeat(self, x, topk_batch_indices, dim):
        if isinstance(x, np.ndarray):
            shape = np.ones(len(list(x.shape)), dtype=int)
            shape[dim] = int(self.batch_size / self.beam_width)
            return np.tile(x.take(topk_batch_indices, axis=dim), tuple(shape.tolist()))
        elif isinstance(x, torch.Tensor):
            shape = torch.ones(len(list(x.shape)), dtype=torch.int)
            shape[dim] = int(self.batch_size / self.beam_width)
            return x.index_select(int(dim), torch.as_tensor(topk_batch_indices, device=x.device), ) \
                .repeat(tuple(shape.numpy().tolist()))
        elif isinstance(x, tuple):
            return tuple((self.general_repeat(t, topk_batch_indices, dim) for t in x))
        elif isinstance(x, list):
            return list((self.general_repeat(t, topk_batch_indices, dim) for t in x))

    def generate_one_measure(self, measure_idx, n_measures, notes, residuals_m21, last_note_net,
                             total_offset, hidden):
        all_workers_done = False
        measure_done = torch.zeros([self.batch_size], dtype=torch.uint8, device=self.device)
        measure_not_done = 1 - measure_done
        duration_in_measure_for_debug = np.zeros([self.batch_size], dtype=Fraction)

        while not all_workers_done:
            prev_residuals = copy.deepcopy(residuals_m21)
            duration_in_measure_for_debug, hidden, last_note_net, measure_done, measure_not_done, \
            notes, residuals_m21, total_offset, all_workers_done = self.generate_one_note(
                duration_in_measure_for_debug,
                hidden, last_note_net,
                measure_done, measure_idx,
                measure_not_done,
                n_measures, notes, residuals_m21,
                total_offset)

            # update self.notes_net only in rows that aren't done yet
            last_dur_m21 = notes[-1][:, DURATION_IDX_IN_NOTE]
            last_dur_nonzero_mask = torch.as_tensor(last_dur_m21.astype(np.float32), device=self.notes_net.device) != 0
            not_residual_mask = torch.as_tensor(prev_residuals[:, 1].astype(np.float32),
                                                device=self.notes_net.device) == 0
            update_mask = (last_dur_nonzero_mask.unsqueeze(-1)) & (not_residual_mask.unsqueeze(-1))

            self.notes_net = torch.where(update_mask,
                                         torch.cat((self.notes_net[1:], last_note_net)),
                                         self.notes_net)

            self.notes_net = self.notes_net.contiguous()
            if self.score_inference is not None:
                self.score_inference.update(self.notes_net, update_mask)

            if self.beam_search == 'note' and len(notes) % self.beam_depth == 0:
                n = np.array(notes)
                if self.score_inference is not None:
                    topk_batch_indices, top_score = self.score_inference.get_topk_batch_indices_from_notes(
                        self.notes_net)
                    self.notes_net = self.general_repeat(self.notes_net, topk_batch_indices, dim=1)
                else:
                    topk_batch_indices, top_score = get_topk_batch_indices_from_notes(n, self.beam_width)

                # select k notes and repeat batch_size/beam_width times:
                notes = list(self.general_repeat(n, topk_batch_indices, dim=1))
                residuals_m21 = self.general_repeat(residuals_m21, topk_batch_indices, dim=0)
                last_note_net = self.general_repeat(last_note_net, topk_batch_indices, 1)
                total_offset = self.general_repeat(total_offset, topk_batch_indices, 0)
                hidden = self.general_repeat(hidden, topk_batch_indices, 1)
                duration_in_measure_for_debug = self.general_repeat(duration_in_measure_for_debug, topk_batch_indices,
                                                                    0)
                measure_done = self.general_repeat(measure_done, topk_batch_indices, 0)
                measure_not_done = 1 - measure_done
            else:
                top_score = 0

        return last_note_net, total_offset, hidden, residuals_m21, notes, top_score

    def generate_one_note(self, duration_in_measure_for_debug, hidden, last_note_net, measure_done, measure_idx,
                          measure_not_done, n_measures, notes, residuals_m21, total_offset):
        # predict next note
        enforce_pitch = None
        enforce_dur = None

        new_hidden, new_dur_m21, new_dur_net, new_note_log_prob, new_pitch, new_tie = self.get_new_note(
            hidden, last_note_net, total_offset, enforce_pitch, enforce_dur)
        residual_exists_mask = self.handle_residual(measure_done, new_dur_m21, new_pitch, new_tie,
                                                    residuals_m21)

        # calculate temporary stats to determine if a measure ended and whether we need to split a note
        new_dur_m21[measure_done.nonzero().cpu().numpy()] = 0
        new_note_log_prob[measure_done.nonzero().cpu()] = 0
        new_note_log_prob[residual_exists_mask.nonzero().cpu()] = 0
        total_offset = total_offset + new_dur_m21
        bar_offset = total_offset % 4
        new_offset = torch.as_tensor((self.fraction_2_float(bar_offset) * 12), dtype=torch.long,
                                     device=self.device)

        # check for bar crosses
        end_at_end_bar = (0 == bar_offset)
        cross_end_bar_mask = ((bar_offset <= ((total_offset - new_dur_m21) % 4)) & (bar_offset != 0)) \
                             | (new_dur_m21 > 4)
        cross_end_bar = cross_end_bar_mask.nonzero()[0]
        at_second_half_of_bar = ((bar_offset >= 2) + cross_end_bar_mask + end_at_end_bar) > 0

        # handle notes that crossed end of bar
        if cross_end_bar.size != 0:
            previous_bar_offset = ((bar_offset[cross_end_bar] - new_dur_m21[cross_end_bar]) % 4)
            duration_until_end_bar = 4 - previous_bar_offset
            residual_duration_m21 = new_dur_m21[cross_end_bar] - duration_until_end_bar
            new_dur_m21[cross_end_bar] = duration_until_end_bar
            new_tie[cross_end_bar] = tie_2_value['start']
            residuals_m21[cross_end_bar] = np.concatenate((new_pitch[cross_end_bar].cpu().numpy(),
                                                           residual_duration_m21[:, np.newaxis],
                                                           new_note_log_prob[cross_end_bar][:, np.newaxis]),
                                                          axis=1)
            total_offset[cross_end_bar] = total_offset[cross_end_bar] - residual_duration_m21
        duration_in_measure_for_debug = duration_in_measure_for_debug + new_dur_m21
        # create new note for the network for next generation
        last_note_in_measure_mask = torch.as_tensor((end_at_end_bar | cross_end_bar_mask).astype(np.long),
                                                    dtype=torch.long, device=self.device)
        next_chord = [self.chords[measure_idx % self.head_len][c] if last_note_in_measure_mask[ind] == 0 else
                      self.chords[(measure_idx + 1) % self.head_len][0] for ind, c in
                      enumerate(at_second_half_of_bar)]
        root, scale_pitches, chord_pitches, chord_idx = \
            chord_2_vec_on_tensor(next_chord, device=self.device)
        new_notes_for_net = torch.cat((new_pitch, new_dur_net,
                                       new_offset.unsqueeze(1),
                                       root.unsqueeze(1), scale_pitches,
                                       chord_pitches, chord_idx.unsqueeze(1)),
                                      dim=1)
        # update last note only where measure is not done and finished handling residuals
        update_last_note_batch_mask = (measure_not_done * (1 - residual_exists_mask.byte()))
        update_last_note_mask = update_last_note_batch_mask.unsqueeze(-1).expand(self.batch_size,
                                                                                 NOTE_VECTOR_SIZE).byte()
        last_note_net = torch.where(update_last_note_mask, new_notes_for_net,
                                    last_note_net[0]).unsqueeze(0)
        measure_idx_tensor = torch.as_tensor(measure_idx,
                                             dtype=torch.long, device=self.device).expand(self.batch_size)
        new_notes_m21 = np.stack((new_pitch.squeeze().cpu().detach().numpy(),
                                  new_dur_m21,
                                  new_tie.cpu().detach().numpy(),
                                  measure_idx_tensor.cpu().numpy(), new_note_log_prob), axis=1)
        notes.append(new_notes_m21)
        update_hiddens_mask = (measure_not_done.long() + residual_exists_mask) > 0
        for i in range(len(hidden)):
            if hidden[0].shape[0] == new_hidden[0].shape[0]:
                hidden[i][:, update_hiddens_mask.nonzero(), :] = new_hidden[i][:, update_hiddens_mask.nonzero(), :]
            else:
                raise ValueError('number of notes in head ({}) is smaller than the memory size ({}). '
                                 'decrease memory size'.format(hidden[0].shape[0], self.model.mem_len))
        measure_done[torch.as_tensor((end_at_end_bar | cross_end_bar_mask).astype(np.long), dtype=torch.long,
                                     device=self.device).nonzero()] = 1
        measure_not_done = 1 - measure_done
        all_workers_done = measure_done.all()

        return duration_in_measure_for_debug, hidden, last_note_net, measure_done, measure_not_done, \
               notes, residuals_m21, total_offset, all_workers_done

    def get_new_note(self, hidden, last_note_net, total_offset, enforce_pitch=None, enforce_duration=None):
        if isinstance(self.model, ChordPitchesModel):
            output_pitch, output_duration, new_hidden = self.model(last_note_net, hidden)
        else:
            output_pitch, output_duration, new_hidden, _ = self.model._forward(last_note_net, [], hidden)

        if len(output_duration.shape) == 3:
            output_pitch = output_pitch[0]
            output_duration = output_duration[0]

        # uncomment to give a bonus for rest notes
        # output_pitch[:,-2] = output_pitch[:, -2] + 5

        output_duration = self.enforce_triplet_and_small_fraction_completion(output_duration, total_offset)

        # remove probs for notes the sax can't produce
        output_pitch[:, :47] = -1e9
        output_pitch[:, 83:-2] = -1e9
        output_pitch[:, -1] = -1e9  # EOS
        if self.score_model == 'reward':
            output_duration[:, self.score_inference.reward_unsupported_durs] = -1e9

        if not (output_duration != -1e9).any(1).all():
            print('all durs -1e9')

        pitch_probs = F.softmax(output_pitch.squeeze() / self.temperature, -1)
        duration_probs = F.softmax(output_duration.squeeze() / self.temperature, -1)
        pitch_log_probs = F.log_softmax(output_pitch.squeeze() / (self.temperature), -1)
        duration_log_probs = F.log_softmax(output_duration.squeeze() / self.temperature, -1)

        if self.non_stochstic_search:
            _, max_inds_pitch = torch.max(pitch_probs, 1)
            _, max_inds_dur = torch.max(duration_probs, 1)

            new_pitch = max_inds_pitch.unsqueeze(1)
            new_dur_net = max_inds_dur.unsqueeze(1)
        elif self.top_p:
            p = 0.9

            topp_p = self.mask_non_top_p(p, pitch_probs)
            topp_d = self.mask_non_top_p(p, duration_probs)

            new_pitch = torch.distributions.categorical.Categorical(probs=topp_p).sample().unsqueeze(-1)
            new_dur_net = torch.distributions.categorical.Categorical(probs=topp_d).sample().unsqueeze(-1)
        else:
            new_pitch = torch.multinomial(pitch_probs, 1)
            new_dur_net = torch.multinomial(duration_probs, 1)

        if self.score_model is not None and len(
                set(self.score_inference.reward_unsupported_durs).intersection(new_dur_net)) > 0:
            print('PROBLEM!!')

        if enforce_pitch is not None:
            new_pitch = enforce_pitch.expand(self.batch_size, 1)
        if enforce_duration is not None:
            new_dur_net = torch.tensor(enforce_duration, dtype=torch.long, device=self.device).expand(self.batch_size,
                                                                                                      1)
        assert 129 not in new_pitch

        new_dur_m21 = self.converter.ind_2_dur_vec(new_dur_net.squeeze())
        new_tie = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        new_pitch_log_prob = torch.gather(pitch_log_probs, 1, new_pitch).detach().cpu().numpy()
        new_dur_log_prob = torch.gather(duration_log_probs, 1, new_dur_net).detach().cpu().numpy()
        new_note_log_prob = (new_pitch_log_prob + new_dur_log_prob).squeeze() + 0.00001
        return new_hidden, new_dur_m21, new_dur_net, new_note_log_prob, new_pitch, new_tie

    def mask_non_top_p(self, p, probs):
        sorted = torch.sort(probs, dim=1, descending=True)
        cumsum = sorted.values.cumsum(dim=1)
        mask = torch.sign(torch.roll(cumsum <= p, 1) + (cumsum <= p))
        for i in range(mask.shape[0]):
            if len(mask[i].nonzero()) == 0:
                mask[i, 0] = 1
        mask = mask.long() * 2 - 1
        topp_idx = sorted.indices * mask.long()
        topp_idx[topp_idx < 0] = -1
        masked_probs = torch.zeros_like(probs)
        for b in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                if topp_idx[b, j] == -1:
                    break
                masked_probs[b, topp_idx[b, j]] = probs[b, topp_idx[b, j]]
        return masked_probs

    def handle_residual(self, measure_done, new_dur_m21, new_pitch, new_tie, residuals_m21):
        residual_exists_mask = (residuals_m21[:, DURATION_IDX_IN_NOTE] > 0) & (1 - measure_done.cpu().numpy())
        residual_exists_indices = residual_exists_mask.nonzero()[0]
        residual_exists_mask = torch.as_tensor(residual_exists_mask,
                                               dtype=torch.long, device=self.device)
        at_least_one_residual_exists = residual_exists_mask.sum() != 0

        if at_least_one_residual_exists:
            new_pitch[residual_exists_indices] = self.object_to_tensor(
                np.asarray(residuals_m21[residual_exists_indices, PITCH_IDX_IN_NOTE]),
                int, new_pitch.dtype).unsqueeze(1)
            new_dur_m21[residual_exists_indices] = residuals_m21[residual_exists_indices, DURATION_IDX_IN_NOTE]

            ties_with_residual = new_tie[residual_exists_indices]
            residuals_larger_than_4_mask = self.object_to_tensor(
                np.asarray(new_dur_m21[residual_exists_indices]).astype(float),
                float, torch.float) > 4
            continue_tensor = torch.ones_like(ties_with_residual).fill_(tie_2_value['continue'])
            end_tensor = torch.ones_like(ties_with_residual).fill_(tie_2_value['stop'])
            new_tie[residual_exists_indices] = torch.where(residuals_larger_than_4_mask,
                                                           continue_tensor,
                                                           end_tensor)

        measure_not_done_idxs = (1 - measure_done).nonzero()
        residuals_m21[measure_not_done_idxs.cpu().numpy(), DURATION_IDX_IN_NOTE] = 0
        return residual_exists_mask

    def enforce_triplet_and_small_fraction_completion(self, output_duration, total_offset):
        # TRIPLETS
        offset_is_triplet = np.asarray([(Fraction(t).denominator / 3) % 1 == 0 for t in total_offset])
        if offset_is_triplet.nonzero()[0].size > 0:
            for i in offset_is_triplet.nonzero()[0]:
                output_duration[i, self.get_non_tuplets_idxs_()] = -1e9
        no_whole_quarter_left_in_measure = np.asarray([t % 4 > 3 for t in total_offset])
        cancel_triplets = (no_whole_quarter_left_in_measure * (1 - offset_is_triplet)).nonzero()[0]
        if cancel_triplets.size > 0:
            for i in cancel_triplets:
                output_duration[i, self.get_tuplets_idxs_()] = -1e9

        # # SMALL FRACTIONS
        # # offset == 1 is one quarter.
        # offset_is_small_fraction = np.asarray([(Fraction(t).denominator / 4) % 1 == 0 and Fraction(t).denominator > 4 for t in total_offset])
        # no_whole_quarter_left_in_measure = np.asarray([t % 4 > 3.5 for t in total_offset])
        # force_small_fractions = (no_whole_quarter_left_in_measure * offset_is_small_fraction).nonzero()[0]
        # if force_small_fractions.size > 0:
        #     for i in force_small_fractions:
        #         if i not in offset_is_triplet.nonzero()[0]:
        #             # if (4 - (total_offset[i] % 4)) < 0.25 and self.score_model and (
        #             #         4 - (total_offset[i] % 4)) not in self.score_inference.reward_supported_durs:
        #             left_in_measure = 4 - (total_offset[i] % 4)
        #             if not (self.score_model and left_in_measure in self.score_inference.reward_supported_durs):
        #                 output_duration[i, self.get_non_x_dur_idxs_(left_in_measure)] = -1e9
        #             else:
        #                 output_duration[i, self.get_non_small_durs_idxs_()] = -1e9
        #         # output_duration[i, self.get_non_x_dur_idxs_(4 - (total_offset[i] % 4))] = -1e9
        # if not (output_duration != -1e9).any(1).all():
        #     print('all durs -1e9')
        return output_duration

    def reset_history(self):
        if isinstance(self.model, ChordPitchesModel):
            self.initial_hidden = self.model.init_hidden(batch_size=self.batch_size)
        else:
            self.initial_hidden = self.model.init_mems()
