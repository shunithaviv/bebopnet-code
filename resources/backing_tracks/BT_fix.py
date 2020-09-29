import os
import time


def bt_fix():
    d = '/home/shunit/backing_tracks_to_fix'
    # folders = [os.path.join(d, o) for o in os.listdir(d)
    #            if os.path.isdir(os.path.join(d, o))]
    # for folder in folders:
    files = [x for x in os.listdir(d) if x.endswith('WAV')]
    # s = 'resources/backing_tracks/active_learning/0.07silence.wav'
    for file in files:
        filepath = os.path.join(d, file[:-4])
        print(filepath)
        # os.system('sox ' + filepath + '.WAV ' + filepath + '_s.WAV trim 00:00:00.02')
        # os.system('sox ' + s + ' ' + filepath + '.WAV ' + filepath + '_s.WAV')
        # time.sleep(1)
        os.system('lame -V2 ' + filepath + '.WAV ' + filepath + '_out.mp3 >/dev/null 2>&1')
        time.sleep(1)
        os.system('sox -v 2 ' + filepath + '_out.mp3 ' + filepath + '.mp3 rate 48k')
        time.sleep(1)


if __name__ == '__main__':
    bt_fix()
