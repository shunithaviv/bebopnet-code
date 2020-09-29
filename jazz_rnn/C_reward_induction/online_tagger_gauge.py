"""
When running this code with pycharm:
- Edit configurations...
- check the checkbox Emulate terminal in output console
"""
"""
images of gauge:
Copyright (C) 2018 FireEye, Inc., created by Andrew Shay. All Rights Reserved.
"""

import datetime
import tkinter
import pyglet

pyglet.options['audio'] = ('openal', 'pulse')
import pyglet.media

import PIL.Image, PIL.ImageTk
from PIL import Image
import argparse
import random
import sys
import os
import music21 as m21
import glob
import pickle

# Song first word to full name: improvisation mp3 should start in the first word of song.
song_name_dict = {'Blue': 'Blue Bossa',
                  'Just': 'Just Friends',
                  'There': 'There Will Never Be Another You',
                  'All': 'All The Things You Are',
                  'Moose': 'Moose The Mooche',
                  'Summertime': 'Summertime',
                  'Ive': 'I''ve Got Rhythm',
                  'Billies': 'Billie''s Bounce',
                  'BlueMonk': 'Blue Monk',
                  'Cheese': 'Cheese Cake',
                  'Confirmation': 'Confirmation',
                  'A': 'A Foggy Day',
                  'Black': 'Black Orpheus',
                  'Chega': 'Chega De Saudade',
                  'Dvarim': 'Lakachta Et Yadi',
                  'Fly': 'Fly me to the moon',
                  'Four': 'Four Brothers',
                  'Giant': 'Giant steps',
                  'Green': 'Green Dolphin',
                  'How': 'How High The Moon',
                  'It': 'It dont mean a thing',
                  'My': 'My Love',
                  'Over': 'Over the rainbow',
                  'Recorda': 'Recorda Me',
                  'Simple': 'Simple Songs',
                  'Well': 'Well you neednt'
                  }

standard_head_mp3_dir = os.path.join('resources', 'standard_heads_mp3')
song_head_mp3_dict = {'Blue': os.path.join(standard_head_mp3_dir, 'Blue_bossa.mp3'),
                      'Just': os.path.join(standard_head_mp3_dir, 'Just_friends.mp3'),
                      'There': os.path.join(standard_head_mp3_dir, 'There_Will_Never_Be_Another_You.mp3'),
                      'All': os.path.join(standard_head_mp3_dir, 'All_The_Things_You_Are.mp3'),
                      'Moose': os.path.join(standard_head_mp3_dir, 'Moose_The_Mooche.mp3'),
                      'Summertime': os.path.join(standard_head_mp3_dir, 'Summertime.mp3'),
                      'Ive': os.path.join(standard_head_mp3_dir, 'I_Got_Rhythm.mp3'),
                      'Billies': os.path.join(standard_head_mp3_dir, 'Billies_Bounce.mp3'),
                      'BlueMonk': os.path.join(standard_head_mp3_dir, 'Blue_Monk.mp3'),
                      'Cheese': os.path.join(standard_head_mp3_dir, 'Cheese_Cake.mp3'),
                      'Confirmation': os.path.join(standard_head_mp3_dir, 'Confirmation.mp3'),
                      'A': os.path.join(standard_head_mp3_dir, 'A_Foggy_Day.mp3'),
                      'Black': os.path.join(standard_head_mp3_dir, 'Black_Orpheus.mp3'),
                      'Chega': os.path.join(standard_head_mp3_dir, 'Chega_De_Saudade.mp3'),
                      'Dvarim': os.path.join(standard_head_mp3_dir, 'Dvarim.mp3'),
                      'Fly': os.path.join(standard_head_mp3_dir, 'Fly_me_to_the_moon.mp3'),
                      'Four': os.path.join(standard_head_mp3_dir, 'Four_Brothers.mp3'),
                      'Giant': os.path.join(standard_head_mp3_dir, 'Giant_steps.mp3'),
                      'Green': os.path.join(standard_head_mp3_dir, 'Green_Dolphin.mp3'),
                      'How': os.path.join(standard_head_mp3_dir, 'How_High_The_Moon.mp3'),
                      'It': os.path.join(standard_head_mp3_dir, 'It_dont_mean_a_thing.mp3'),
                      'My': os.path.join(standard_head_mp3_dir, 'My_Love.mp3'),
                      'Over': os.path.join(standard_head_mp3_dir, 'Over_the_rainbow.mp3'),
                      'Recorda': os.path.join(standard_head_mp3_dir, 'Recorda_Me.mp3'),
                      'Simple': os.path.join(standard_head_mp3_dir, 'Simple_Songs.mp3'),
                      'Well': os.path.join(standard_head_mp3_dir, 'Well_you_neednt.mp3'),
                      }


class UserLabels:
    def __init__(self, username, repeats, save_dir='results/user_labels'):
        self.tagger = username
        self.song_labels = {}
        self.repeats = repeats
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)
        user_tags_exists = len(
            glob.glob(os.path.join(save_dir, 'user_labels_{}reps_{}*'.format(self.repeats, username)))) > 0
        if user_tags_exists:
            file_list = glob.glob(os.path.join(save_dir, 'user_labels_{}reps_{}*'.format(self.repeats, username)))
            file_list.sort()
            file = file_list[-1]

            print('Previous tags found!\n Do you wish to load previous tags? (yes/no)')
            user_input = input().lower()
            if user_input == 'yes' or user_input == 'y':
                with open(file, 'rb') as f:
                    self.song_labels = pickle.load(f)
                print('Loaded previous tags!')
        self.save_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

    def add_song(self, filename, idx):
        song_name = self.name_from_file(filename, idx)
        self.song_labels[song_name] = SongLabels(song_name)

    def reset_song(self, filename, idx):
        self.add_song(filename, idx)

    def update_label(self, filename, idx, label, offset):
        song_name = self.name_from_file(filename, idx)
        self.song_labels[song_name].update(label, offset)

    def save(self):
        pkl_name = os.path.join(self.save_dir,
                                "user_labels_{}reps_{}_{}.pkl".format(self.repeats, self.tagger, self.save_time))
        with open(pkl_name, 'wb') as f:
            pickle.dump(self.song_labels, f)

    def last_offset(self, filename, idx):
        song_name = self.name_from_file(filename, idx)
        try:
            return self.song_labels[song_name].label_offsets[-1]
        except IndexError:
            return 0

    def get_pkl_name(self):
        return os.path.join(self.save_dir,
                            "user_labels_{}reps_{}_{}.pkl".format(self.repeats, self.tagger, self.save_time))

    @staticmethod
    def name_from_file(filename, idx):
        song_name = os.path.basename(filename).replace('.mp3', '')
        return '{}_{}'.format(song_name, idx)


class SongLabels:
    def __init__(self, song_name):
        self.song_file_name = song_name
        self.labels = []
        self.label_offsets = []  # in quarter_length

    def update(self, label, offset):
        self.labels.append(label)
        self.label_offsets.append(offset)


def extract_bpm(song):
    for elem in m21.converter.parse(song).parts[0].flat.getElementsByOffset(0.0):
        if isinstance(elem, m21.tempo.MetronomeMark):
            return elem.number
    raise ValueError('Couldn''t find metronome mark at offset 0 of xml {}'.format(song))


class MusicGauge:
    def __init__(self, window, window_title, user_labels, song, idx):
        self.percent = 50
        image_path = "jazz_rnn/utils/python_gauge/gauge_images/50_gauge.jpg"
        self.move = ''

        self.user_labels = user_labels
        self.user_labels.add_song(song, idx)
        self.song = song
        self.idx = idx

        self.before_start_playing = True
        self.paused = False
        self.playing_head = False
        song_swing = song.replace('.mp3', '_swing.mp3')
        self.init_player(song_swing)

        bpm = extract_bpm(song.replace('mp3', 'xml'))
        bps = bpm / 60
        self.spb = 1 / bps  # seconds per beat

        self.window = window
        self.window.title(window_title)

        # Load an image using OpenCV
        self.image = Image.open(image_path)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.width, self.height = self.image.size

        # Resize image
        self.height, self.width = self.height // 2, self.width // 2

        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=self.image.resize((self.width, self.height)))

        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.bind('<space>', self.play_or_pause)
        self.window.bind('<Right>', self.right_key)
        self.window.bind('<Left>', self.left_key)
        self.window.bind('<KeyRelease>', self.stop)
        self.window.bind('<Delete>', self.restart)
        self.window.bind('<Escape>', self.quit)
        self.window.bind('h', self.play_head)
        self.window.bind('H', self.play_head)
        self.window.focus()

        self.animate()

        print('*****************************************************')
        print('Playing {}'.format(song_name_dict[song.split('/')[-1].split('_')[0]]))
        print('press h to play or stop a reminder of the head')
        print()
        print('press SPACE to play or pause the song')
        print('press  ESC  to quit and save the current song')
        print('press  DEL  to restart tagging the current song')

        self.window.mainloop()

    def init_player(self, song_path_mp3):

        # if this fails, you should install avbin.
        # follow instructions from here: https://stackoverflow.com/a/45833386/3693922
        # sudo sh ./install-avbin-linux-x86-64-v10
        pyglet_song = pyglet.media.load(song_path_mp3, streaming=False)
        self.player = pyglet.media.Player()
        self.player.queue(pyglet_song)

    def animate(self):
        # For each of the (← →) keys currently being pressed move in the corresponding direction
        if self.move == 'right':
            self.right()
        elif self.move == 'left':
            self.left()
        elif self.move == 'quit':
            self.window.destroy()
            return

        self.canvas.update()

        offset = int((self.player.time - 0.3) / self.spb)
        one_beat_interval = offset >= self.user_labels.last_offset(self.song, self.idx) + 1
        if offset > 0 and not self.paused and one_beat_interval:
            self.user_labels.update_label(self.song, self.idx, self.percent, offset)

        # This method calls itself again and again after a delay (80 ms in this case)
        self.window.after(80, self.animate)

    def play_head(self, e):
        self.player.pause()
        if self.playing_head:
            self.init_player(self.song)
            self.restart(0)
            self.playing_head = False
        else:
            self.init_player(song_head_mp3_dict[self.song.split('/')[-1].split('_')[0]])
            self.player.play()
            self.playing_head = True

    def play_or_pause(self, e):
        if self.player.playing:
            self.player.pause()
        else:
            self.player.play()

    def restart(self, e):
        self.player.pause()
        self.player.seek(0)

        self.percent = 50
        self.update_image()
        self.move = ''
        self.user_labels.reset_song(self.song, self.idx)
        self.before_start_playing = True
        self.paused = False

    def stop(self, e):
        self.move = ''

    def right_key(self, e):
        self.move = 'right'

    def left_key(self, e):
        self.move = 'left'

    def quit(self, e):
        self.move = 'quit'
        self.user_labels.save()
        self.player.pause()

    def right(self):
        if self.percent < 100:
            self.percent += 25
        self.update_image()

    def left(self):
        if self.percent > 0:
            self.percent -= 25
        self.update_image()

    # Callback for the "Blur" button
    def update_image(self):
        image_path = "jazz_rnn/utils/python_gauge/gauge_images/{}_gauge.jpg".format(self.percent)
        self.image = Image.open(image_path)
        self.photo = PIL.ImageTk.PhotoImage(image=self.image.resize((self.width, self.height)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)


def tagger(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default='./results/samples',
                        help='dir for input mp3 files')
    parser.add_argument('--repeats', type=int, default=1,
                        help='number of times to tag each song')
    parser.add_argument('--labels_save_dir', type=str, default='results/user_labels',
                        help='dir for saving user labels')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--name', type=str, default='',
                        help='enter name by script')
    args = parser.parse_args(args)

    if args.seed:
        random.seed(args.seed)

    if args.name == '':
        print('What''s your name? (type first name and then press ENTER)')
        username = input()
    else:
        username = args.name

    user_labels = UserLabels(username, args.repeats, save_dir=args.labels_save_dir)

    train_song_list = [os.path.join(args.dir, 'train', x) for x in os.listdir(os.path.join(args.dir, 'train')) if
                       x.endswith('mp3')]
    try:
        test_song_list = [os.path.join(args.dir, 'test', x) for x in os.listdir(os.path.join(args.dir, 'test')) if
                          x.endswith('mp3')]
    except FileNotFoundError:
        print('No Test Folder')
        test_song_list = []
    all_songs = train_song_list + test_song_list
    # all_songs = [os.path.join(args.dir, x) for x in os.listdir(args.dir) if x.endswith('mp3')]

    all_songs = [s for s in all_songs if not 'swing' in s]

    c = 0
    for i in range(args.repeats):
        for song in all_songs:
            if UserLabels.name_from_file(song, i) in user_labels.song_labels.keys():
                continue
            c += 1
    print('{} songs left to tag!'.format(c))

    for i in range(args.repeats):
        random.shuffle(all_songs)
        for song in all_songs:
            if UserLabels.name_from_file(song, i) in user_labels.song_labels.keys():
                continue
            MusicGauge(tkinter.Tk(), "Preference Meter", user_labels, song, i)

    return user_labels.get_pkl_name()


if __name__ == '__main__':
    tagger(sys.argv[1:])
