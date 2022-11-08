import myfunctions as fn
import sys
import librosa
import numpy as np

conv_stft = lambda freq, frame: \
  np.sum(y_stft[y_stft.shape[0]*freq//33:y_stft.shape[0]*(freq+1)//33, frame])
list_to_bit = lambda list:np.sum([bit * 2**i for i, bit in enumerate(list)])

db = fn.Database("database/music_light.db")

m = db.load_Music_by_ID(0)
mono_y = librosa.to_mono(m.y)
y_chroma = librosa.feature.chroma_stft(librosa.to_mono(m.y))
print(y_chroma.shape)
frame_result = np.ndarray((y_chroma.shape[0], y_chroma.shape[1]), dtype='bool')

for frame in range(y_chroma.shape[1]-1):
	print(f"{frame: >5}: ", end="")
	for chroma in range(y_chroma.shape[0]-1):
		frame_result[chroma, frame] = \
			y_chroma[chroma][frame+1] - y_chroma[chroma+1][frame+1]  -\
				y_chroma[chroma][frame] + y_chroma[chroma+1][frame]> 0
		print(int(frame_result[chroma, frame]), end="")
		if chroma % 4 == 3:
			print(end=" ")
	frame_result[11, frame] = \
		y_chroma[11][frame+1] - y_chroma[0][frame+1] -\
			y_chroma[11][frame] + y_chroma[0][frame]> 0
	print(int(frame_result[11, frame]), end="")
	print(f" : 0x{list_to_bit(frame_result[:, frame]):0>3x}")
print(f"size: ({y_chroma.shape[0]}, {y_chroma.shape[1]-1})")