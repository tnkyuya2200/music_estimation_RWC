import myfunctions as fn
import sys
import librosa
import numpy as np
from bitarray import bitarray

list_to_bit = lambda list:np.sum([int(bit * 2**i) for i, bit in enumerate(list)])
nbin_beat_sum = lambda nbin, beat: np.sum(y_cqt[nbin][beat:beat+1])
db = fn.Database("database/music_light2.db")

m = db.load_Music_by_ID(1)
mono_y = librosa.to_mono(m.y)
y_cqt = librosa.cqt(librosa.to_mono(m.y))
"""
frame_result = np.ndarray((y_cqt.shape[0]-1, y_cqt.shape[1]), dtype='bool')
result = np.ndarray((y_cqt.shape[1]), dtype='int32')
for frame in range(y_cqt.shape[1]-1):
    print(f"{frame: >5}:", end="")
    for nbin in range(y_cqt.shape[0]-1):
        frame_result[nbin, frame] = \
            y_cqt[nbin][frame+1] - y_cqt[nbin+1][frame+1] -\
                y_cqt[nbin][frame] + y_cqt[nbin+1][frame] > 0
        if nbin % 4 == 0:
            print(end=" ")
        print(int(frame_result[nbin, frame]), end="")
    #print(" : 0x{:0>32}".format(list_to_bit(frame_result[:, frame])))
    print()
print(f"size: ({y_cqt.shape[0]-1}, {y_cqt.shape[1]-1})")
"""
frame_result = np.ndarray((len(m.beats[:-2]), y_cqt.shape[1]), dtype='bool')
result = np.ndarray((len(m.beats[:-2])), dtype='object')
for beat_idx, beat in enumerate(m.beats[:-2]):
    beat_result = bitarray(y_cqt.shape[0]-1)
    for nbin in range(1, y_cqt.shape[0]):
        beat_result[nbin-1] = \
            nbin_beat_sum(nbin, beat) + nbin_beat_sum(nbin-1, beat+1) \
                - nbin_beat_sum(nbin, beat+1) - nbin_beat_sum(nbin-1, beat) > 0
        if nbin%4 == 0:
            print(end=" ")
        print(int(beat_result[nbin-1]), end="")
    result[beat_idx] = beat_result
    print()
print(f"size: ({y_cqt.shape[0]-1}, {y_cqt.shape[1]-1})")