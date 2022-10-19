import myfunctions as fn
import sys
import librosa
import numpy as np

conv_stft = lambda freq, frame: \
  np.sum(y_stft[y_stft.shape[0]*freq//33:y_stft.shape[0]*(freq+1)//33, frame])
list_to_bit = lambda list:np.sum([bit * 2**i for i, bit in enumerate(list)])

db = fn.Database("database/music_light.db")

m = db.load_Music_by_ID(7)
mono_y = librosa.to_mono(m.y)
y_stft = np.abs(librosa.stft(librosa.to_mono(m.y)))

result = np.ndarray((32, y_stft.shape[1]), dtype='bool')
for frame in range(y_stft.shape[1]-1):
  print(frame, end=":\t")
  for freq in range(0, 32):
    result[freq, frame] = \
      conv_stft(freq, frame+1) - conv_stft(freq+1, frame+1) -\
         conv_stft(freq, frame) + conv_stft(freq+1, frame) > 0
    print(int(result[freq, frame]), end=", ")
  print(": 0x{:0>8x}".format(list_to_bit(result[:, frame])))