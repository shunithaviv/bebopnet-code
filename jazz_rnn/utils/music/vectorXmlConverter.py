from functools import lru_cache

from bidict import bidict
import numpy as np
import torch
import music21 as m21


class VectorXmlConverter:
    def __init__(self, durations):
        self.bidict = bidict()
        for d_ind, d in enumerate(sorted(list(durations))):
            self.bidict.put(d, d_ind)

    @lru_cache(maxsize=80)
    def dur_2_ind(self, duration):
        return self.bidict[duration]

    @lru_cache(maxsize=80)
    def ind_2_dur(self, vec):
        return self.bidict.inv[vec]

    def max_durations(self):
        return len(self.bidict)

    def ind_2_dur_vec(self, ind_tensor):
        ind_np = ind_tensor.cpu().detach().numpy()
        durs = []
        for i in ind_np:
            durs.append(self.ind_2_dur(i))
        return np.asarray(durs)

    def dur_2_ind_vec(self, dur_tensor):
        dur_np = dur_tensor
        inds = []
        for i in dur_np:
            inds.append(self.dur_2_ind(i))
        return inds


REST_IDX = 128
N_NOTES = 13
N_NOTES_NO_REST = N_NOTES - 1
N_PITCHES = 129
N_PITCHES_NO_REST = 128

NOTE_VECTOR_SIZE = 31


def create_note(pitch_idx, duration, tie=None):
    if pitch_idx == REST_IDX:
        n = m21.note.Rest(quarterLength=duration)
        assert (n is not None)
        return n
    else:
        n = m21.note.Note(midi=pitch_idx, quarterLength=duration)
        assert (n is not None)
        if tie:
            t = m21.tie.Tie(tie)
            n.tie = t
        return n


note_char_2_ind = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E-': 3, 'E': 4,
                   'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A-': 8,
                   'A': 9, 'B-': 10, 'B': 11}

ind_2_note_char = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                   6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

tie_none = 0
tie_start = 1
tie_continue = 2
tie_stop = 3

tie_2_value = {'start': tie_start, 'continue': tie_continue, 'stop': tie_stop}


@lru_cache(maxsize=256)
def ensure_4_notes(chord):
    chord_kind = chord.chordKind

    if 'major-minor' in chord_kind or 'minor-major' in chord_kind:
        return m21.harmony.ChordSymbol(kind='minor-major-seventh',
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    if chord_kind == 'major':
        return m21.harmony.ChordSymbol(kind='dominant',
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    if len(chord.pitches) < 4:
        return m21.harmony.ChordSymbol(kind=chord_kind + '-seventh',
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    if 'ninth' in chord_kind:
        return m21.harmony.ChordSymbol(kind=chord_kind.replace('ninth', 'seventh'),
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    if '11th' in chord_kind:
        return m21.harmony.ChordSymbol(kind=chord_kind.replace('11th', 'seventh'),
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    if '13th' in chord_kind:
        return m21.harmony.ChordSymbol(kind=chord_kind.replace('13th', 'seventh'),
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)
    else:
        return m21.harmony.ChordSymbol(kind=chord_kind,
                                       root=chord.root().simplifyEnharmonic(mostCommon=True).name)


@lru_cache(maxsize=256)
def chord_2_vec(input_chord, relative_pitch=None, song=''):
    scale_notes = np.zeros(N_NOTES, dtype=int)
    chord_notes = np.zeros(N_NOTES, dtype=int)
    if not input_chord:
        scale_notes[-1] = chord_notes[-1] = 1
        root = np.array([N_NOTES_NO_REST])
        chord_idx = N_CHORD_TYPES - 1
    else:
        assert isinstance(input_chord, m21.harmony.ChordSymbol)

        if len(input_chord.pitches) != 4:
            chord = ensure_4_notes(input_chord)
        else:
            chord = input_chord

        if len(chord.pitches) > 4 and chord.chordKind != 'suspended-fourth':
            raise ValueError('not handled correctly. chord: {}'.format(chord))

        scale_notes = np.zeros(N_NOTES, dtype=int)
        chord_notes = np.zeros(N_NOTES, dtype=int)
        root_ind = note_char_2_ind[chord.root().simplifyEnharmonic(mostCommon=True).name]
        root = root_ind
        scale_pitches = get_scale_pitches_from_chord(chord, song)
        scale_pitches = list(set(scale_pitches))
        scale_pitch_indices = [note_char_2_ind[p.simplifyEnharmonic(mostCommon=True).name] %
                               N_NOTES_NO_REST for p in scale_pitches]
        chord_pitch_indices = [note_char_2_ind[p.simplifyEnharmonic(mostCommon=True).name] %
                               N_NOTES_NO_REST for p in chord.pitches]
        if (chord.chordKind == 'suspended-fourth' or chord.chordKind == 'suspended-fourth-seventh') and len(
                chord_pitch_indices) != 4:  # Fix m21 bug with 'suspended-fourth-seventh' chords
            chord_pitch_indices = [(root + p) % N_NOTES_NO_REST for p in [0, 5, 7, 10]]
        if relative_pitch is not None and relative_pitch != REST_IDX:
            chord_pitch_indices = [note_char_2_ind[p.simplifyEnharmonic(mostCommon=True).name] - relative_pitch
                                   for p in chord.pitches]
            rank = relative_pitch % N_NOTES_NO_REST
            chord_pitch_indices = [(i - rank) % N_NOTES_NO_REST for i in chord_pitch_indices]
        scale_notes[scale_pitch_indices] = 1
        chord_notes[chord_pitch_indices[:4]] = 1
        assert sum(chord_notes) == 4, str(song) + ', ' + str(chord)
        chord_idx = chord_2_idx(chord)

    return root, scale_notes.tolist(), chord_notes.tolist(), chord_idx


def chord_2_vec_on_tensor(chords, device=None):
    results = []
    for c in chords:
        results.append(chord_2_vec(c))

    list_to_tensor = lambda x: torch.tensor(np.array(x), dtype=torch.long, device=device)
    root_list, scale_pitches_list, chord_pitches_list, chord_idx_list = map(list_to_tensor, zip(*results))

    return root_list, scale_pitches_list, chord_pitches_list, chord_idx_list


_chord_2_idx = {'major': 0, 'major-seventh': 0, 'major-sixth': 0,
                'minor': 1, 'minor-seventh': 1, 'minor-sixth': 1, 'minor-ninth': 1, 'minor-11th': 1,
                'mM7': 2, 'minor-major-seventh': 2,
                'm7b5': 3, 'half-diminished-seventh': 3, 'half-diminished': 3,
                'dominant-seventh': 4, 'suspended-fourth': 4, 'suspended-fourth-seventh': 4,
                'dominant-ninth': 4, 'dominant-13th': 4,
                'dominant': 4, 'major-minor': 4, 'major-minor-seventh': 4,
                'augmented': 5, 'augmented-seventh': 5,
                'diminished': 6, 'diminished-seventh': 6
                }
N_CHORD_TYPES = 8
N_CHORD_TYPES_NO_BLANK = N_CHORD_TYPES - 1
NO_CHORD_IDX = N_CHORD_TYPES - 1


def chord_2_idx(chord):
    chord_kind = chord.chordKind
    return _chord_2_idx[chord_kind]


# for every chords, which pitches are suitable
def get_scale_pitches_from_chord(chord, song=''):
    chord_kind = chord.chordKind
    root = chord.root()
    if 'major' == chord_kind or 'major-seventh' == chord_kind or 'major-sixth' == chord_kind:
        intervals = np.array([0, 2, 2, 1, 2, 2, 2])
    elif 'minor' == chord_kind or 'minor-seventh' == chord_kind or 'minor-sixth' == chord_kind \
            or 'minor-ninth' == chord_kind or 'minor-11th' == chord_kind:
        intervals = np.array([0, 2, 1, 2, 2, 1, 2])
    elif 'mM7' == chord_kind:
        intervals = np.array([0, 2, 1, 2, 2, 1, 3])
    elif 'm7b5' == chord_kind or 'half-diminished-seventh' == chord_kind \
            or 'half-diminished' == chord_kind:
        intervals = np.array([0, 1, 2, 2, 1, 2, 2])
    elif 'dominant-seventh' == chord_kind or 'suspended-fourth' == chord_kind \
            or 'suspended-fourth-seventh' == chord_kind \
            or 'dominant-ninth' == chord_kind \
            or 'dominant-13th' == chord_kind or 'dominant' == chord_kind \
            or 'major-minor' == chord_kind or 'major-minor-seventh' == chord_kind:
        intervals = np.array([0, 2, 2, 1, 2, 2, 1])
    elif 'augmented' == chord_kind or 'augmented-seventh' == chord_kind:
        intervals = np.array([0, 2, 2, 2, 2, 1, 2])
    elif 'diminished' == chord_kind or 'diminished-seventh' == chord_kind:
        intervals = np.array([0, 2, 2, 1, 1, 2, 2])
    elif 'minor-major' == chord_kind or 'major-minor' == chord_kind or \
            'minor-major-seventh' == chord_kind or 'major-minor-seventh' == chord_kind:
        intervals = np.array([0, 2, 1, 2, 2, 2, 2])
    else:
        raise ValueError('unrecognized chord: ' + chord_kind + ' in ' + song)
    absolute_intervals = np.cumsum(intervals)
    return [m21.pitch.Pitch(root.midi + absolute_interval) for absolute_interval in absolute_intervals]


def input_2_groups(input_var, bptt, batch_size):
    pitch = torch.squeeze(input_var[:, :, 0]).contiguous().view(bptt, -1)
    duration = torch.squeeze(input_var[:, :, 1]).contiguous().view(bptt, -1)
    offset = torch.squeeze(input_var[:, :, 2]).contiguous().view(bptt, -1)
    root = torch.squeeze(input_var[:, :, 3]).contiguous().view(bptt, -1)
    scale_pitches = input_var[:, :, 4:17].contiguous().view(bptt, batch_size, -1)
    chord_pitches = input_var[:, :, 17:30].contiguous().view(bptt, batch_size, -1)

    chord_idx = torch.squeeze(input_var[:, :, -1])
    rank = pitch_2_rank(pitch, root)
    octave = pitch_2_octave(pitch)
    return pitch, duration, offset, root, scale_pitches, chord_pitches, chord_idx, rank, octave


def tie_idx_2_value(tie_idx):
    try:
        tie_list = [None] + list(tie_2_value)
        return tie_list[tie_idx]
    except:
        raise ValueError('Invalid tie value {}'.format(tie_idx))


def pitch_2_rank(pitch, root):
    return (pitch - root) % N_NOTES_NO_REST


def pitch_2_octave(pitch):
    return pitch / N_NOTES_NO_REST
