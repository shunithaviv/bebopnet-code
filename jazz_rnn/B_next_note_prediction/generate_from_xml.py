import datetime
import json
import argparse
import pickle
import sys
import os
import time
import random
import glob

import numpy as np
import music21 as m21
import lxml.etree as le
import torch

from jazz_rnn.B_next_note_prediction.generation_utils import song_params_dict, pop_bt2silence
from jazz_rnn.B_next_note_prediction.music_generator import MusicGenerator
from jazz_rnn.utils.music_utils import notes_to_stream, notes_to_swing_notes
from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM


def generate_from_xml(args):
    parser = argparse.ArgumentParser(description='PyTorch Jazz Language Model')
    try:
        m21.environment.set('musicxmlPath', '/usr/bin/musescore')
    except m21.environment.UserSettingsException:
        print('no musescore')
        pass

    gen_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S_%f")

    input_parser = parser.add_argument_group('Input Parameters')
    input_parser.add_argument('--model_dir', type=str,
                              help='model and converter(pickled data corpus) directory to use')
    input_parser.add_argument('--checkpoint', type=str,
                              default='model.pt',
                              help='model checkpoint to use in model_dir')
    input_parser.add_argument('--score_model', type=str,
                              default='',
                              help='path for score model that was created with sequence comparisons or "harmony" for'
                                   'harmony based score')
    input_parser.add_argument('--threshold', type=float, default='0.0',
                              help='threshold for scores')

    # XML, BT and MIDIvol are overridden by song
    song_parser = parser.add_argument_group('Song Parameters')
    song_parser.add_argument('--song', type=str, default='fly', choices=song_params_dict.keys(),
                             help='name of song to generate (shortcut for xml and bt definition')
    song_parser.add_argument('--xml_', type=str, default='',
                             help='xml (overrides song)')

    # Output related parameters
    output_parser = parser.add_argument_group('Output Parameters')
    output_parser.add_argument('--add_to_outf', type=str, default='',
                               help='output file for generated musicXml')
    output_parser.add_argument('--save_dir', type=str, default='./results/samples/',
                               help='dir for output files')
    output_parser.add_argument('--create_mp3', type=int, default='1',
                               help='generate mp3 file with backing track')
    output_parser.add_argument('--remove_head_from_mp3', action='store_true', default=0,
                               help='remove head melody from mp3, play only improvisation')

    generation_parser = parser.add_argument_group('Generation Parameters')
    generation_parser.add_argument('--seed', type=int, default=None,
                                   help='random seed')
    generation_parser.add_argument('--no-cuda', action='store_true', default=0,
                                   help='use CUDA')
    generation_parser.add_argument('--temperature', type=float, default=1,
                                   help='temperature - higher will increase diversity')
    generation_parser.add_argument('--num_measures', type=int, default=64,
                                   help='number of measures to generate')
    generation_parser.add_argument('--num_heads', type=int, default=1,
                                   help='number of heads to generate (if not 0, overrides num_measures)')
    generation_parser.add_argument('--batch_size', type=int, default=2,
                                   help='number of parallel generation for every measure')
    generation_parser.add_argument('--beam_search', type=str, default='measure', choices=['', 'note', 'measure'],
                                   help='beam search type: note|measure|'' ''')
    generation_parser.add_argument('--beam_width', type=int, default=2,
                                   help='beam width for beam search')
    generation_parser.add_argument('--beam_depth', type=int, default=1,
                                   help='beam depth for beam search (units of beam_search')
    generation_parser.add_argument('--non-stochstic-search', action='store_true', default=0,
                                   help='whether to sample using max or softmax')
    generation_parser.add_argument('--top-p', action='store_true', default=1,
                                   help='sample using top-p (nucleus sampling). https://arxiv.org/pdf/1904.09751.pdf')
    generation_parser.add_argument('--verbose', action='store_true', default=0,
                                   help='verbose option for xml to mp3 process')

    args = parser.parse_args(args)
    args.cuda = not args.no_cuda
    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    if (args.batch_size / args.beam_width) % 1 > 0:
        parser.error("--beam width has to be a divider of batch size")

    set_rnd_seed(args)

    set_args_by_song_name(args, gen_time)
    args.remove_head_from_mp3 = args.remove_head_from_mp3 or (
            args.back_track.split('/')[-1] in pop_bt2silence.keys())

    if args.xml_ != '':
        args.xml = args.xml_

    args.verbose_ext = ' 2>/dev/null'
    if args.verbose:
        args.verbose_ext = ''

    converter_path = os.path.join(args.model_dir, 'converter_and_duration.pkl')
    with open(converter_path, 'rb') as input_file:
        converter = pickle.load(input_file)

    # generate model
    is_transformer = len(glob.glob(os.path.join(args.model_dir, 'scripts', 'mem_transformer.py'))) != 0
    if is_transformer:
        with open(os.path.join(args.model_dir, 'args.json'), 'r') as f_params:
            kwargs = json.load(f_params)
        if args.song == 'giant':
            kwargs['mem_len'] = 54
        with open(os.path.join(args.model_dir, args.checkpoint), 'rb') as f:
            model = MemTransformerLM(**kwargs)
            model_path = os.path.join(args.model_dir, args.checkpoint)
            model.load_state_dict(torch.load(model_path))
        model.converter = converter
    else:
        with open(os.path.join(args.model_dir, args.checkpoint), 'rb') as f:
            if args.no_cuda:
                model = torch.load(f, map_location=lambda storage, loc: storage)
            else:
                model = torch.load(f)

    if args.cuda:
        model.cuda()

    model.eval()

    generator = MusicGenerator(model, converter, batch_size=args.batch_size,
                               beam_search=args.beam_search,
                               beam_width=args.beam_width,
                               beam_depth=args.beam_depth,
                               non_stochastic_search=args.non_stochstic_search,
                               top_p=args.top_p,
                               temperature=args.temperature,
                               score_model=args.score_model, threshold=args.threshold,
                               ensemble=True, song=args.song, no_head=args.remove_head_from_mp3)

    generator.init_stream(args.xml)

    if args.num_heads != 0:
        args.num_measures = args.num_heads * generator.head_len

    notes, top_likelihood = generator.generate_measures(args.num_measures)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.score_model != '' and args.beam_search != '':
        if top_likelihood < 0:
            args.outf = args.outf + '_avgNegScore_' + str(np.abs(top_likelihood))
        else:
            args.outf = args.outf + '_avgScore_' + str(np.abs(top_likelihood))
    else:
        if top_likelihood < 0:
            args.outf = '{}_neglikelihood_{:.2f}'.format(args.outf, np.abs(top_likelihood))
        else:
            args.outf = '{}_likelihood_{:.2f}'.format(args.outf, np.abs(top_likelihood))

    notes_swing = notes_to_swing_notes(notes[:, 0, :])

    stream = notes_to_stream(notes[:, 0, :], generator.stream, generator.chords, generator.head_len,
                             args.remove_head_from_mp3, head_early_start=generator.early_start)
    stream_swing = notes_to_stream(notes_swing, generator.stream, generator.chords, generator.head_len,
                                   args.remove_head_from_mp3, head_early_start=generator.early_start)
    if not args.remove_head_from_mp3:
        stream_no_head = notes_to_stream(notes[:, 0, :], generator.stream, generator.chords, generator.head_len,
                                         remove_head=True, head_early_start=generator.early_start)
        # Create output files:
        xml_converter = m21.converter.subConverters.ConverterMusicXML()
        try:
            xml_converter.write(stream_no_head, 'musicxml', args.outf + '_no_head.xml')
        except m21.musicxml.m21ToXml.MusicXMLExportException:
            print('cannot create no_head version')
            return generator
        except m21.duration.DurationException:
            print('cannot create no_head version (dur_exp)')
            return generator

    # Create output files:
    xml_converter = m21.converter.subConverters.ConverterMusicXML()
    try:
        xml_converter.write(stream, 'musicxml', args.outf + '.xml')
    except m21.musicxml.m21ToXml.MusicXMLExportException:
        print('failed generating xml for {}'.format(args.outf))
        return generator

    # show in musescore- open Musescore
    # stream.show('musicxml')

    # remove chords from xml:
    os.system('cp ' + args.outf + '.xml ' + args.outf + '_with_chords.xml')
    with open(args.outf + '.xml', 'rb') as f:
        doc = le.parse(f)
        root = doc.getroot()
        for elem in root.iter('harmony'):
            parent = elem.getparent()
            parent.remove(elem)
        doc.write(args.outf + '.xml')

    # create music stream without chords;
    music_stream_out = m21.converter.parse(args.outf + '.xml').parts[0]
    music_stream_out.autoSort = False

    create_midi(args, music_stream_out)

    if args.create_mp3:
        create_mp3(args)

    generator.filename = args.outf
    generator.score = top_likelihood

    # MP3 with swing
    args.outf = args.outf + '_swing'
    # Create output files:
    xml_converter = m21.converter.subConverters.ConverterMusicXML()
    try:
        xml_converter.write(stream_swing, 'musicxml', args.outf + '.xml')
    except m21.musicxml.m21ToXml.MusicXMLExportException:
        print('cannot create swing version')
        return generator

    # remove chords from xml:
    os.system('cp ' + args.outf + '.xml ' + args.outf + '_with_chords.xml')
    with open(args.outf + '.xml', 'rb') as f:
        doc = le.parse(f)
        root = doc.getroot()
        for elem in root.iter('harmony'):
            parent = elem.getparent()
            parent.remove(elem)
        doc.write(args.outf + '.xml')
    # MP3 with swing
    music_stream_out = m21.converter.parse(args.outf + '.xml').parts[0]
    music_stream_out.autoSort = False
    create_midi(args, music_stream_out)
    if args.create_mp3:
        create_mp3(args)

    return generator


def create_midi(args, music_stream_out):
    mf = m21.midi.translate.streamToMidiFile(music_stream_out)
    mf.open(args.outf + '.mid', 'wb')
    mf.write()
    mf.close()


def create_mp3(args):
    # generate mp3 midi
    os.system(
        ('timidity --preserve-silence {0}.mid -Ow -o -' + args.verbose_ext + ' | lame - -b 64 -h {0}_initial.mp3 >/dev/null 2>&1').format(
            args.outf))

    if args.back_track.split('/')[-1] in pop_bt2silence.keys():
        # add silence to despacito and increase volume
        silence = pop_bt2silence[args.back_track.split('/')[-1]]
        # increase volume
        os.system('sox -v 5 {o}_initial.mp3 {o}_n.mp3 rate 48k'.format(o=args.outf) + args.verbose_ext)
        os.system('sox {0} {1}_n.mp3 {1}_notes.mp3 rate 48k'.format(silence, args.outf) + args.verbose_ext)
    else:
        # increase volume
        os.system('sox -v 4 {o}_initial.mp3 {o}_notes.mp3 rate 48k'.format(o=args.outf) + args.verbose_ext)

    # merge with backing track
    os.system('sox -m {0}_notes.mp3 {1} {0}_full.mp3'.format(args.outf, args.back_track) + args.verbose_ext)
    # adjust volume
    os.system('sox -v 1.5 {0}_full.mp3 {0}.mp3'.format(args.outf) + args.verbose_ext)

    time.sleep(1)
    # remove tmp files
    os.system('rm {0}_notes.mp3'.format(args.outf))
    os.system('rm {0}_initial.mp3'.format(args.outf))
    os.system('rm {0}_full.mp3'.format(args.outf))
    os.system('rm {0}.mid'.format(args.outf))
    if args.back_track.split('/')[-1] in pop_bt2silence:
        os.system('rm {0}_n.mp3'.format(args.outf))


def set_args_by_song_name(args, gen_time):
    song_params = song_params_dict[args.song]
    if args.song == '':
        args.xml = args.xml_
        args.back_track = './backing_tracks/48/{}'.format(song_params.back_track)
        return 0

    args.xml = './resources/xmls/{}'.format(song_params.xml)
    args.back_track = './resources/backing_tracks/standards/2heads/{}'.format(song_params.back_track)
    args.outf = os.path.join(args.save_dir, '{}_{}'.format(song_params.outf, gen_time))
    if args.add_to_outf != '':
        args.outf = args.outf + '_{}'.format(args.add_to_outf)


def set_rnd_seed(args):
    if not (args.seed is None):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            if args.no_cuda == 1:
                print("WARNING: You have a CUDA device, so you should probably run without --no-cuda")
            else:
                torch.cuda.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    generate_from_xml(sys.argv[1:])
