import sys
import argparse
import glob
import math
import os
import copy
import pickle
from functools import partial
from itertools import filterfalse
from fractions import Fraction

import torch.multiprocessing as mp
from jazz_rnn.utils.music.vectorXmlConverter import *
from jazz_rnn.C_reward_induction.online_tagger_gauge import SongLabels

# A vector consists of:
# [0]       1 int  - pitch can get value from 0 to 127
# [1]       1 int  - duration - a float, where 1.0 = 1 quarter note. 0.5 = eighth note. 2.0 = half note.
# [2]       1 int  - offset - location of the note within the bar in 8th length 0 to 16
# [3]       1 int  - chord root - from 0 to 13, the root of the chord
# [4:17]  13 ints - scale pitches - indicators of participating pitches
# [18:31] 13 ints - chord pitches - indicators of participating pitches
# [31]      1 int  - chord idx - type of chord (Major, minor...)
REST_SYMBOL = 128
EOS_SYMBOL = 129
EOS_VECTOR = [EOS_SYMBOL] + [0] * 30
EOS_REWARD_VECTOR = [EOS_SYMBOL] + [0] * 31
LEGAL_DENOMINATORS = [1, 2, 3, 4, 6]


class NoChordError(Exception):
    pass


def extract_data_from_xml(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=str, default='resources/dataset',
                        help='directory holding the xml files')
    parser.add_argument('--labels_file', type=str, default='',
                        help='file holding the user labels')
    parser.add_argument('--out_dir', type=str,
                        default='results/dataset_pkls/')
    parser.add_argument('--num_processes', type=int, default=11,
                        help='How many processes to use. Default=11')
    parser.add_argument('--exclude', nargs='+',
                        help='Which directories to ignore')
    parser.add_argument('--cached_converter', type=str, default='',
                        help='use the converter from previous run')
    parser.add_argument('--check_time_signature', action='store_true',
                        help='move songs with different time signature to other folder')
    parser.add_argument('--no_test', action='store_true',
                        help='don''t process test data')
    parser.add_argument('--no_eos', action='store_true',
                        help='Gather data for lstm training (no eos)')

    args = parser.parse_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.labels_file:
        args.reward_induction = True
        args.out_dir = os.path.join(args.out_dir, 'ri')
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.reward_induction = False

    if args.reward_induction:
        train_songs = glob.glob(args.xml_dir + "/train/xml_with_chords/*.xml")
        test_songs = glob.glob(args.xml_dir + "/test/xml_with_chords/*.xml")
    else:
        train_songs = glob.glob(args.xml_dir + "/train/*/*.xml")
        test_songs = glob.glob(args.xml_dir + "/test/*/*.xml")

    if args.exclude is not None:
        def in_exclude_list(x): return any([y in x for y in args.exclude])

        train_songs[:] = filterfalse(in_exclude_list, train_songs)
        test_songs[:] = filterfalse(in_exclude_list, test_songs)

    all_songs = train_songs + test_songs

    song_labels_dict = {}
    if args.reward_induction:
        with open(args.labels_file, 'rb') as f:
            song_labels_dict = pickle.load(f)

    if args.check_time_signature:
        check_time_signature(all_songs, args.xml_dir)

    if args.cached_converter == '':
        durations, _ = get_all_durations(all_songs)
        converter = VectorXmlConverter(durations)
    else:
        print('loaded converter')
        with open(args.cached_converter, 'rb') as input_converter:
            converter = pickle.load(input_converter)
            durations = pickle.load(input_converter)

    n_proc = min(args.num_processes, len(train_songs))
    if n_proc > 1:
        pool = mp.Pool(processes=n_proc)
        partial_extract_vectors = partial(extract_vectors, ri=args.reward_induction,
                                          song_labels_dict=song_labels_dict,
                                          converter=converter, no_eos=args.no_eos)

        train_results = pool.map(partial_extract_vectors, train_songs)
        if not args.no_test:
            test_results = pool.map(partial_extract_vectors, test_songs)
        pool.close()
        pool.join()
    else:  # used for debug
        train_results = extract_vectors(song=train_songs[0], ri=args.reward_induction,
                                        song_labels_dict=song_labels_dict,
                                        converter=converter, no_eos=args.no_eos)

    train_data = results_2_dict(train_results, train_songs)
    if not args.no_test:
        test_data = results_2_dict(test_results, test_songs)

    train_data = remove_consecutive_rest_vars(train_data, converter, args.reward_induction, no_eos=args.no_eos)
    if not args.no_test:
        test_data = remove_consecutive_rest_vars(test_data, converter, args.reward_induction, no_eos=args.no_eos)

    def dict_2_np(x):
        return {k: np.array(v) for k, v in x.items() if np.array(v).shape[0] != 0}

    train_data = dict_2_np(train_data)
    if not args.no_test:
        test_data = dict_2_np(test_data)

    # train_data = transpose_data_dict(pitches, train_data)
    # test_data = transpose_data_dict(pitches, test_data)

    os.makedirs(args.out_dir, exist_ok=True)
    print('saving results to {}'.format(args.out_dir))

    with open(os.path.join(args.out_dir, 'converter_and_duration.pkl'), 'wb') as fp:
        pickle.dump(converter, fp)
        pickle.dump(durations, fp)
    with open(os.path.join(args.out_dir, 'train.pkl'), 'wb') as fp:
        pickle.dump(train_data, fp)
    if not args.no_test:
        with open(os.path.join(args.out_dir, 'val.pkl'), 'wb') as fp:
            pickle.dump(test_data, fp)

    print('All done!')
    return train_data


def transpose_data_dict(pitches, data_dict):
    new_data_dict = {}
    for song_name, data in data_dict.items():
        tranposed_data_list = []
        for i, p in enumerate(pitches):
            transposed_data = transpose_data(p, data)
            tranposed_data_list.append(transposed_data)
        new_data_dict[song_name] = np.concatenate(tranposed_data_list, axis=0)
    return new_data_dict


def transpose_data(p, data):
    tranposed_data = copy.deepcopy(data)
    not_rest_or_eos_mask = (data[:, 0] != REST_SYMBOL) & (data[:, 0] != EOS_SYMBOL)
    tranposed_data[not_rest_or_eos_mask, 0] = data[not_rest_or_eos_mask, 0] + p
    tranposed_data[:, 3] = (data[:, 3] + p) % 12
    tranposed_data[:, 4:16] = np.roll(data[:, 4:16], shift=p, axis=1)
    tranposed_data[:, 17:29] = np.roll(data[:, 17:29], shift=p, axis=1)
    return tranposed_data


def results_2_dict(results, songs):
    results_dict = {}
    for song, result in zip(songs, results):
        song_name = os.path.basename(song).replace(' ', '_')
        results_dict[song_name] = result
    return results_dict


def get_all_durations(songs):
    note_with_tie_dur = None
    durations = set()
    dur_hist = {}
    illegal_duration_set = {}
    small_duration_set_6 = {}
    small_duration_set_8 = {}

    for song in songs:
        s = m21.converter.parse(song)
        for n in s.flat.getElementsByClass(['Note', 'Rest']):
            quarter_length = n.duration.quarterLength
            if n.tie is not None:
                if n.tie.type == 'start':
                    if note_with_tie_dur is not None:
                        note_with_tie_dur = note_with_tie_dur + Fraction(quarter_length)
                    else:
                        note_with_tie_dur = Fraction(quarter_length)
                    continue

                elif n.tie.type == 'continue':
                    assert note_with_tie_dur is not None, str(song) + str(n)
                    note_with_tie_dur = note_with_tie_dur + Fraction(quarter_length)
                    continue

                elif n.tie.type == 'stop':
                    assert note_with_tie_dur is not None, str(song) + str(n)
                    note_with_tie_dur = note_with_tie_dur + Fraction(quarter_length)
                    quarter_length = note_with_tie_dur
                    note_with_tie_dur = None

                else:
                    raise ValueError('Tie value invalid')

            if quarter_length:
                durations.add(quarter_length if isinstance(quarter_length, Fraction)
                              else Fraction(str(quarter_length)))
                try:
                    dur_hist[str(float(quarter_length))] += 1
                except KeyError:
                    dur_hist[str(float(quarter_length))] = 1

            if quarter_length < Fraction(1, 6):
                try:
                    small_duration_set_6[song] += 1
                except KeyError:
                    small_duration_set_6[song] = 1

            if quarter_length < Fraction(1, 8):
                try:
                    small_duration_set_8[song] += 1
                except KeyError:
                    small_duration_set_8[song] = 1

    print('Illegal Durations: (dur: #occurences)')
    print(illegal_duration_set)
    print('The following songs have more than 60 notes with duration < 1/6: ')
    for k, v in small_duration_set_6.items():
        if v > 60:
            print(v, k)

    print('number of durations < 1/8: ', sum(small_duration_set_8.values()), ', number of durations < 1/6: ',
          sum(small_duration_set_6.values()))

    return sorted(durations), dur_hist


def add_to_db(converter, database, pitch, duration, offset, chord, ri, label):
    # check if fraction is legal, add to db
    # if Fraction(duration).denominator in LEGAL_DENOMINATORS:
    try:
        bar_offset = offset % 4
        bar_offset_in_48 = int(math.floor(bar_offset * 12))
        root, scale_notes, chord_notes, chord_idx = chord_2_vec(chord)
        if ri:
            assert label is not None
            new_data = [pitch,
                        converter.bidict[duration],
                        bar_offset_in_48,
                        root] + scale_notes + chord_notes + [chord_idx] + [label]
        else:
            new_data = [pitch,
                        converter.bidict[duration],
                        bar_offset_in_48,
                        root] + scale_notes + chord_notes + [chord_idx]
        if chord_idx == NO_CHORD_IDX:
            raise NoChordError()
        assert sum(chord_notes) == 4

        database.append(new_data)
    except KeyError:
        print('illegal duration. skipping...')
        pass
    return database


def add_eos(database, index=None, ri=False):
    if ri:
        eos_vector = EOS_REWARD_VECTOR
    else:
        eos_vector = EOS_VECTOR
    if not index:
        database.append(eos_vector)
    else:
        database.insert(index, eos_vector)


def remove_consecutive_rest_vars(data_dict, converter, ri, no_eos=False):

    if no_eos:
        def is_rest_measure(ind):
            return data[ind][0] == REST_SYMBOL and data[ind][1] == converter.dur_2_ind(Fraction(4, 1))

        for _, data in data_dict.items():
            rest_measure_indices = []
            for i in range(len(data) - 1):
                if is_rest_measure(i) and is_rest_measure(i + 1):
                    rest_measure_indices.append(i)
            for i in sorted(rest_measure_indices, reverse=True):
                del data[i]
        return data_dict

    def is_rest_measure(data, ind):
        return data[ind][0] == REST_SYMBOL and data[ind][1] == converter.dur_2_ind(Fraction(4, 1))

    new_data_dict = {}
    for name, data in data_dict.items():
        i = 0
        new_data_dict[name] = []
        while i < len(data) - 1:
            if is_rest_measure(data, i) and is_rest_measure(data, i + 1):
                j = 2
                while is_rest_measure(data, i + j):
                    j += 1
                i += j
                del new_data_dict[name][-1]
                if not no_eos:
                    add_eos(new_data_dict[name], ri=ri)
            else:
                new_data_dict[name].append(data[i])
                i += 1
    return new_data_dict


def extract_vectors(song, ri, song_labels_dict, converter, no_eos=False):
    data = []
    if ri:
        song_key = os.path.basename(song).replace('_with_chords.xml', '_0')
        song_labels = song_labels_dict[song_key].labels
    else:
        song_labels = []
    s = m21.converter.parse(song).parts.stream()
    current_chord = None
    note_with_tie = None
    last_note = None
    total_offset = 0
    try:
        for n in s.flat.notesAndRests:
            label = None
            if isinstance(n, m21.chord.Chord):
                if not isinstance(n, m21.harmony.ChordSymbol):
                    n = m21.harmony.ChordSymbol(n.pitches[0].name)
                current_chord = n

                _, _, _, chord_idx = chord_2_vec(current_chord, song=song)
                assert chord_idx != NO_CHORD_IDX, str(song)

                if last_note is None:
                    last_note = m21.note.Note(current_chord.pitches[0])
                continue

            quarter_length = n.duration.quarterLength
            pitch = n.pitch.midi if issubclass(n.__class__, m21.note.NotRest) else REST_SYMBOL

            duration = quarter_length if isinstance(quarter_length, Fraction) else Fraction(str(quarter_length))
            if duration == 0:  # grace note
                continue
            offset = n.offset
            total_offset += duration

            if n.tie is not None:
                tie_type = n.tie.type
                if tie_type == 'start':
                    if note_with_tie is not None:
                        note_with_tie[1] = note_with_tie[1] + duration
                    else:
                        note_with_tie = [pitch, duration, offset, current_chord]

                elif tie_type == 'continue':
                    note_with_tie[1] = note_with_tie[1] + duration

                elif tie_type == 'stop':
                    note_with_tie[1] = note_with_tie[1] + duration
                    if ri:
                        label_idx = int(total_offset) + int((total_offset % 1) > 0)
                        try:
                            label = song_labels[label_idx - 1]
                        except IndexError:
                            print('finished labels before song {}'.format(song))
                            break
                    data = add_to_db(converter, data, *note_with_tie, ri, label)
                    note_with_tie = None

                else:
                    raise ValueError('Tie value invalid (' + str(n.tie.type) + ') in ' + str(song))
            else:
                if current_chord is None:
                    print('chord is none in', song)
                if ri:
                    label_idx = int(total_offset) + int((total_offset % 1) > 0)
                    try:
                        label = song_labels[label_idx - 1]
                    except IndexError:
                        print('finished labels before song {}'.format(song))
                        break
                data = add_to_db(converter, data, pitch, duration, offset, current_chord, ri, label)
    except NoChordError:
        print('missing a chord in {}'.format(song))
        exit(1)
    if not no_eos:
        add_eos(data, ri=ri)
    return data


def extract_chords_from_xml(xml_file):
    s = m21.converter.parse(xml_file)
    measures = s.parts[0].getElementsByClass(m21.stream.Measure)

    chords = []
    for i, m in enumerate(measures):
        c = m.getElementsByClass(m21.harmony.ChordSymbol)
        if len(c) == 0:
            if i > 0:
                chords.append(chords[-1])
        elif len(c) == 1:
            chords.append([c[0], c[0]])
        elif len(c) == 2:
            chords.append([c[0], c[1]])

    return chords


def check_time_signature(songs, xml_dir):
    for song in songs:
        s = m21.converter.parse(song)
        if s.parts[0]._getTimeSignatureForBeat().numerator != 4:
            raise ValueError('Illegal time signature in ' + song)
            # os.system('mv ' + song + ' ' + xml_dir + 'removed/')


if __name__ == '__main__':
    extract_data_from_xml(sys.argv[1:])
