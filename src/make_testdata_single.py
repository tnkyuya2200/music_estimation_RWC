import myfunctions as fn
import numpy as np
import soundfile as sf
import librosa
import sys, os
import random
def main():
	db = fn.Database(sys.argv[1])
	dir_name = sys.argv[2]
	ID = int(sys.argv[3])
	m = db.load_Music_by_ID(ID)

	sf.write(os.path.join(dir_name, "raw", str(ID)+".wav"), m.y.T, m.sr)

	#TODO: noise test data
	y_length = m.y.shape[1]
	s = np.random.random(size=y_length)*2 - 1
	noise_change = round(random.uniform(0.1, 0.3), 2)

	noise = np.array((s*noise_change, s*noise_change))
	y_with_noise = librosa.util.normalize(m.y + noise, axis=1)
	sf.write(os.path.join(dir_name, "noise", str(ID)+".noise_"+str(noise_change)+".wav"), y_with_noise.T, m.sr)

	#TODO: snipped test data
	length_time = random.randrange(60, np.min((180, round(librosa.samples_to_time(y_length, m.sr)*0.8))))
	length_samples = librosa.time_to_samples(length_time, m.sr)
	max_start_samples = y_length - length_samples
	start_samples = random.randrange(0, max_start_samples)
	end_samples = start_samples + length_samples
	y_snipped = m.y[:, start_samples:end_samples]
	sf.write(os.path.join(dir_name, "snipped", str(ID)+".snipped_"+str(start_samples)+"_"+str(end_samples)+".wav"), y_snipped.T, m.sr)

	speed_change = round(random.uniform(0.5, 1.5), 2)
	if speed_change > 0.9 and speed_change < 1.1:
		speed_change = round(random.uniform(0.5, 1.5))
	y_speed_change = np.array([librosa.effects.time_stretch(m.y[0,:], speed_change), librosa.effects.time_stretch(m.y[1,:], speed_change)])
	sf.write(os.path.join(dir_name, "speed", str(ID)+".speed_"+str(speed_change)+".wav"), y_speed_change.T, m.sr)

	pitch_change = random.randrange(-5, 5)
	if pitch_change == 0:
		pitch_change = random.randrange(-5, 5)
	y_pitch_change = np.array([librosa.effects.pitch_shift(m.y[0,:], m.sr, pitch_change),librosa.effects.pitch_shift(m.y[1,:], m.sr, pitch_change)])
	sf.write(os.path.join(dir_name, "pitch", str(ID)+".pitch_"+str(pitch_change)+".wav"), y_pitch_change.T, m.sr)
	print(ID, ",", start_samples,",", end_samples,",", noise_change, ",", speed_change, "," , pitch_change)

if __name__ == "__main__":
	main()