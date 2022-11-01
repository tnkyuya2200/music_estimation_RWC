import myfunctions as fn
import sys
import librosa
import numpy as np

conv_stft = lambda n_bin, n_beat: \
  np.sum(y_cqt[n_bin, beats[n_beat]:beats[n_beat+1]])
list_to_bit = lambda list:np.sum([bit * 2**i for i, bit in enumerate(list)])
db = fn.Database("database/music_light.db")

m = db.load_Music_by_ID(1)
mono_y = librosa.to_mono(m.y)
y_cqt = librosa.cqt(librosa.to_mono(m.y))
bpm, beats = librosa.beat.beat_track(y=librosa.to_mono(m.y))

beat_result = np.ndarray((y_cqt.shape[0]-1, y_cqt.shape[1]), dtype='bool')
for n_beat in range(len(beats)-2):
	print(f"{n_beat: >5}:", end="")
	for n_bin in range(y_cqt.shape[0]-1):
		beat_result[n_bin, n_beat] = \
			conv_stft(n_bin, n_beat+1) - conv_stft(n_bin+1, n_beat+1) -\
				conv_stft(n_bin, n_beat) + conv_stft(n_bin+1, n_beat) > 0
		if n_bin % 4 == 0:
			print(end=" ")
		print(int(beat_result[n_bin, n_beat]), end="")
	print()
print(f"size: ({y_cqt.shape[0]-1}, {len(beats)-2})")