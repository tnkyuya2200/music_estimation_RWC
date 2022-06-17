#coding:utf-8
import numpy as np
import librosa
from tqdm import tqdm
from spleeter.separator import Separator
from collections import OrderedDict
import json
import sqlite3
import io
import warnings
import copy
import os
import codecs
import tensorflow as tf
warnings.simplefilter('ignore')

tf.compat.v1.disable_eager_execution()
def make_chorddict():
	chord_dic = OrderedDict()
	one_third = 1.0/3
#major chords
	chord_dic["C"] = [one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0,0]
	chord_dic["Db"] = [0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0]
	chord_dic["D"] = [0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0]
	chord_dic["Eb"] = [0,0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0]
	chord_dic["E"] = [0,0,0,0, one_third, 0,0,0, one_third, 0,0, one_third]
	chord_dic["F"] = [one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0,0]
	chord_dic["Gb"] = [0, one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0]
	chord_dic["G"] = [0,0, one_third, 0,0,0,0, one_third, 0,0,0, one_third]
	chord_dic["Ab"] = [one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0,0]
	chord_dic["A"] = [0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0]
	chord_dic["Bb"] = [0,0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0]
	chord_dic["B"] = [0,0,0, one_third, 0,0, one_third, 0,0,0,0, one_third]
#minor chords
	chord_dic["Cm"] = [one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0,0]
	chord_dic["Dbm"] = [0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0]
	chord_dic["Dm"] = [0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0]
	chord_dic["Ebm"] = [0,0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0]
	chord_dic["Em"] = [0,0,0,0, one_third, 0,0, one_third, 0,0,0, one_third]
	chord_dic["Fm"] = [one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0,0]
	chord_dic["Gbm"] = [0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0]
	chord_dic["Gm"] = [0,0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0]
	chord_dic["Abm"] = [0,0,0, one_third, 0,0,0,0, one_third, 0,0, one_third]
	chord_dic["Am"] = [one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0,0]
	chord_dic["Bbm"] = [0, one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0]
	chord_dic["Bm"] = [0,0, one_third, 0,0,0, one_third, 0,0,0,0, one_third]
	return chord_dic

chord_dic = make_chorddict()

def estimate_chord(chroma, chord_dic=chord_dic):
	maximum = -1
	this_chord = np.zeros(12)
	for chord_index, (name, vector) in enumerate(chord_dic.items()):
		similarity = cos_sim(chroma, vector)
		if similarity > maximum:
			maximum = similarity
			this_chord = vector
	return this_chord

def estimate_chords(chromas, chord_dic=chord_dic):
	result = np.empty(chromas.shape)
	for i in range(chromas.shape[1]):
		result[:,i] = estimate_chord(chromas[:,i], chord_dic)
	return result

def adapt_array(arr):
	"""
	http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
	"""
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())

def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out, allow_pickle=True)

def sep_by_nan(data, threshold=1):
	result = []
	cont = []
	nan_count = 0
	for d in data:
		if np.isnan(d):
			nan_count += 1
		else:
			nan_count = 0
		cont.append(d)
		if nan_count >= threshold:
			nan_count = 0
			while len(cont) != 0 and np.isnan(cont[0]):
				del cont[0]
			result.append(cont[:-threshold])
			cont = []
		while len(cont) != 0 and np.isnan(cont[0]):
			del cont[0]

	result.append(cont)
	result = [x for x in result if len(x) >= threshold]
	return result

def sep_count(data, count=10, threshold=4):
	result = sep_by_nan(data, threshold)
	while len(result) > count:
		threshold += 1
		result = sep_by_nan(data, threshold)
	result = np.array(result, dtype=object)
	return result

def compare_melody(input_melody, database_melody):
	sim = []
	for index_db in range(len(database_melody)):
		db_sample = database_melody[index_db]
		sim_db = [0]
		for index_input in range(len(input_melody)):
			input_sample = input_melody[index_input]
			sim_lag = [0]
			if len(input_sample) > len(db_sample):
				for lag in range(len(input_sample)-len(db_sample)+1):
					diff = []
					both_nan = 0
					for i in range(len(db_sample)):
						diff.append((input_sample[i+lag] - db_sample[i]))
						if np.isnan(input_sample[i+lag]) and np.isnan(db_sample[i]):
							both_nan += 1
					is_nan = np.isnan(diff)
					med = np.median([x for i,x in enumerate(diff) if not(is_nan[i])])
					sim_lag.append((len([i for i, x in enumerate(diff) if abs(x-med)<=0.6])+both_nan)/len(diff))
			else:
				for lag in range(len(db_sample)-len(input_sample)+1):
					diff = []
					both_nan = 0
					for i in range(len(input_sample)):
						diff.append((input_sample[i] - db_sample[i+lag]))
						if np.isnan(input_sample[i]) and np.isnan(db_sample[i+lag]):
							both_nan += 1
					is_nan = np.isnan(diff)
					med = np.median([x for i,x in enumerate(diff) if not(is_nan[i])])
					sim_lag.append((len([i for i, x in enumerate(diff) if abs(x-med)<=0.6])+both_nan)/len(diff))
			sim_db.append(max(sim_lag))
		sim.append(max(sim_db))
	return np.mean(sim)

def cos_sim(v1,v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def corr_cossim(data1, data2):
	sim_i = []
	for i in range(min(data1.shape[1], data2.shape[1])):
		sim_i.append(cos_sim(data1[:,i], data2[:,i]))
	return np.mean(sim_i)

def compare_acc(input_acc, database_acc, separate=64):
	shorter_acc = None
	longer_acc = None
	if input_acc.shape[1] < database_acc.shape[1]:
		shorter_acc = database_acc
		longer_acc = input_acc
	else:
		shorter_acc = input_acc
		longer_acc = database_acc
	sim_i = []
	for i in range(12):
		rolled_shorter_acc = np.roll(shorter_acc, i, axis=0)
		sim = []
		for index_shorter in range(shorter_acc.shape[1]//separate):
			shorter_sample = rolled_shorter_acc[:,index_shorter*separate:min((index_shorter+1)*separate, shorter_acc.shape[1]-1)]
			sim_index = []
			for index_longer in range(longer_acc.shape[1]-separate):
				longer_sample = longer_acc[:,index_longer:index_longer+separate]
				sim_index.append(corr_cossim(shorter_sample, longer_sample))
			sim.append(max(sim_index))
		sim_i.append(np.mean(sim))
	return max(sim_i)
	'''
	sim = []
	for index_shorter in range(shorter_acc.shape[1]//separate):
		sim_index = []
		shorter_sample = shorter_acc[:,index_shorter*separate:min((index_shorter+1)*separate, shorter_acc.shape[1]-1)]
		for index_longer in range(longer_acc.shape[1]-separate):
			longer_sample = longer_acc[:,index_longer:index_longer+separate]
			sim_index.append(corr_cossim(shorter_sample, longer_sample))
		sim.append(max(sim_index))
	return np.mean(sim)
	'''
def compare(input, data):
	sim_melody = compare_melody(input.melody, data.melody)
	#sim_acc = compare_acc(input.acc, data.acc)
	sim_chords = compare_acc(input.chords, data.chords)
	return sim_melody,  sim_chords

def spleeter_4stems_separate(y, sep=20000000):
	result = {"vocals":[], "drums":[], "bass":[], "other":[]}
	separator = Separator("spleeter:4stems", multiprocess=True)
	for i in range(0, y.shape[1], sep):
		prediction = separator.separate(y[:,i:min(i+sep, m.y.shape[1])].T)
		result["vocals"].append(prediction["vocals"].T)
		result["drums"].append(prediction["drums"].T)
		result["bass"].append(prediction["bass"].T)
		result["other"].append(prediction["other"].T)
	return result

def sep_quantize(arr, quantize):
	q_arr = arr.tolist()
	if quantize > 4:
		quantize = quantize // 4
		arr_len = len(arr)
		for index in range(len(arr)-1):
			for i in range(1,quantize):
				q_arr.append(q_arr[index]+(q_arr[index+1]-q_arr[index]) * i//quantize)
		q_arr.sort()
		return np.array(q_arr)
	if quantize < 4:
		quantize = 4 // quantize
		return arr[::quantize]
	return arr

def f0_in_beats(y, sr, beats=None, vocals=None, quantize=8):
	if beats is None:
		_, beats = librosa.beat.beat_track(y=librosa.to_mono(y), sr=sr)
	q_beats = sep_quantize(beats, 8)
	if vocals is None:
		vocals = librosa.util.normalize(librosa.to_mono(spleeter_4stems_separate(y)["vocals"]))
	f0 = (list(map(lambda x:round(x-12), librosa.hz_to_midi(librosa.yin(librosa.to_mono(vocals),fmin=65,fmax=2093,sr=sr)))))
	output = []
	threshold = np.percentile(abs(vocals), 25)
	for index in range(len(q_beats)-1):
		note = np.median(f0[q_beats[index]:q_beats[index+1]])
		if np.median(np.abs(vocals[q_beats[index]*512:q_beats[index+1]*512])) < threshold or note >= 80 :
			output.append(np.nan)
		else:
			output.append(note)
	return np.array(output)

def chroma_in_beats(y, sr, beats=None, acc_wav=None):
	TONES = 12
	if acc_wav is None:
		acc_wav = spleeter_4stems_separate(y)['other']
	if beats is None:
		_, beats = librosa.beat.beat_track(y=librosa.to_mono(y), sr=sr)
	chroma = librosa.feature.chroma_cens(librosa.to_mono(acc_wav), sr)
	sum_chroma = np.zeros((TONES, len(beats)+1))
	beats_count = 0
	for frame_index in range(chroma.shape[1]):
		for i in range(TONES):
			sum_chroma[i, beats_count] += np.abs(chroma[i, frame_index])
		if frame_index in beats:
			beats_count += 1
	return sum_chroma


def compare_all(test_FilePath=None, test_music=None, db_FilePath="music.db"):
	db = Database(db_FilePath)
	if test_FilePath is None:
		test = test_music
	else:
		test = Music()
		test.load_music(test_FilePath)
	test.analyze(4)
	datas = db.load_all()
	result = {}

	test_q2 = copy.deepcopy(test)
	test_q2.analyze(2)

	for x in tqdm(datas, leave=False, position=1):
		result[x.FilePath] = {"sim":{}}
		x_q2 = copy.deepcopy(x)
		x_q2.analyze(2)
		if test.bpm < x.bpm*3/4:
			vocal_sim, chords_sim = compare(test, x_q2)
			result[x.FilePath]["sim"]["vocal"] = vocal_sim
			result[x.FilePath]["sim"]["chords"] = chords_sim
			result[x.FilePath]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
		elif test.bpm > x.bpm*3/2:
			vocal_sim, chords_sim = compare(test_q2, x)
			result[x.FilePath]["sim"]["vocal"] = vocal_sim
			result[x.FilePath]["sim"]["chords"] = chords_sim
			result[x.FilePath]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
		else:
			vocal_sim, chords_sim = compare(test,x)
			result[x.FilePath]["sim"]["vocal"] = vocal_sim
			result[x.FilePath]["sim"]["chords"] = chords_sim
			result[x.FilePath]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
	return result
	"""
	for x in tqdm(datas, leave=False, position=1):
		vocal_sim, _, chords_sim = compare(test,x)
		result[x.FilePath]["sim"]["vocal"].append(vocal_sim)
		result[x.FilePath]["sim"]["chords"].append(chords_sim)
		result[x.FilePath]["sim"]["average"].append(np.mean((vocal_sim, chords_sim)))

	#test_copy = copy.deepcopy(x)
	#test_copy.analyze(8)
	for x in tqdm(datas, leave=False, position=1):
		x_copy = copy.deepcopy(x)
		x_copy.analyze(2)
		#vocal_sim, _, chords_sim = compare(test_copy, x)
		vocal_sim, _, chords_sim = compare(test, x_copy)
		result[x.FilePath]["sim"]["vocal"].append(vocal_sim)
		result[x.FilePath]["sim"]["chords"].append(chords_sim)
		result[x.FilePath]["sim"]["average"].append(np.mean((vocal_sim, chords_sim)))

	test_copy = copy.deepcopy(test)
	test_copy.analyze(2)
	for x in tqdm(datas, leave=False, position=1):
		#x_copy = copy.deepcopy(x)
		#x_copy.analyze(8)
		#vocal_sim, _, chords_sim = compare(test, x_copy)
		vocal_sim, _, chords_sim = compare(test_copy, x)
		result[x.FilePath]["sim"]["vocal"].append(vocal_sim)
		result[x.FilePath]["sim"]["chords"].append(chords_sim)
		result[x.FilePath]["sim"]["average"].append(np.mean((vocal_sim, chords_sim)))

	json_FilePath = os.FilePath.join("result_q_2", os.FilePath.splitext(os.FilePath.basename(test_FilePath))[0])
	file = open(json_FilePath+".json",'w',encoding='utf-8')
	json.dump(result, file, indent=2, ensure_ascii=False)
	file.close()

	output = {}
	for x in datas:
		max, max_index = result[x.FilePath]["sim"]["average"][0], 0
		for i in range(1,3):
			if max < result[x.FilePath]["sim"]["average"][i]:
				max, max_index = result[x.FilePath]["sim"]["average"][i], i
		output[x.FilePath] = {"sim":{}}
		output[x.FilePath]["sim"] = {"vocal": result[x.FilePath]["sim"]["vocal"][max_index],
								"chords":result[x.FilePath]["sim"]["chords"][max_index],
								"average":result[x.FilePath]["sim"]["average"][max_index]}
	return output
	"""

def init_database(con, cur):
	query = """
CREATE TABLE IF NOT EXISTS music(
ID INTEGER PRIMARY KEY,
NO INTEGER,
Composer TEXT,
Composer_Eng TEXT,
Artist TEXT,
Artist_Eng Text,
Title TEXT,
Title_Eng TEXT,
CD TEXT,
Track_No TEXT,
Genre TEXT,
Genre_Eng TEXT,
Sub_Genre TEXT,
Sub_Genre_ENG TEXT,
FileFilePath TEXT,
y array,
sr INTEGER,
beats ARRAY,
bpm INTEGER,
frame_size INTEGER,
quantize INTEGER,
melody ARRAY,
acc ARRAY,
chords ARRAY,
section ARRAY
);
	"""
	con.execute(query)

def load_database(FilePath):
	con = sqlite3.connect(FilePath, detect_types=sqlite3.PARSE_DECLTYPES)
	cur = con.cursor()
	return con, cur

class Database:
	Path = None         # FilePath to file
	con = None          # sqlite3.connect object
	cur = None          # cursor object
	def __init__(self, Path="music.db"):
		sqlite3.register_adapter(np.ndarray, adapt_array)
		sqlite3.register_converter("array", convert_array)
		self.Path = Path
		self.con, self.cur = load_database(Path)
		init_database(self.con, self.cur)
		self.con.isolation_level = None
	def __del__(self):
		self.con.close()
	def selectall(self, col):
		query = "select "+col+" from music;"
		self.cur.execute(query)
		return self.cur.fetchall()
	def selectsingle(self, col):
		query = "select " + col + "from music;"
		self.cur.execute(query)
		return self.cur.fetchone()
	def load_all(self):
		datas = self.selectall("ID, FilePath, sr, beats, bpm, frame_size, quantize, acc, chords")
		musics = []
		for data in datas:
			music = Music()
			music.load_database(data)
			musics.append(music)
		return musics
	def load_single(self):
		data = self.selectsingle("ID, FilePath, sr, beats, bpm, frame_size, quantize, acc, chords")
		music = Music()
		music.load_database(data)
		return music
	def load_ID(self, ID=0):
		query = "select ID, FilePath, sr, beats, bpm, frame_size, quantize, acc, chords from music where ID = ?;"
		self.cur.execute(query, (ID,))
		return self.cur.fetchone()
	def getdbsize(self):
		query = "select count(*) from music;"
		self.cur.execute(query)
		return self.cur.fetchone()


class Music:
	ID = None           # Song ID                               int
	y = None            # wav series                            np.ndarray, shape=(2,samples)
	FilePath = None     # FilePath to file                      str
	sr = 0              # samplerate                            int                          
	beats = None        # frames where beat is                  np.ndarray, shape=(frames,)
	bpm = 0             # beats per minute                      int
	frame_size = 512    # frame size (default 512 samples)      int
	quantize = 4        # how often you get melody notes        int
	predictions = None  # spleeter prediction                   dict
	melody = None       # analyzed cqt_bins in each beats       np.ndarray, shape=(len(beats),)
	acc = None          # analyzed accompaniment in each beat   np.ndarray, shape=(12,len(beats))
	chords = None       # analyzed chords                       np.ndarray, shape=(12,len(beats))
	def load_music(self, FilePath):
		self.ID = 0
		self.FilePath = FilePath
		self.y, self.sr = librosa.load(FilePath, sr=None, mono=False)
		self.y = librosa.util.normalize(self.y, axis=1)
	"""
	def from_json(self, data):
		music_data = json.loads(data)
		FilePath = music_data['FilePath']
		name = music_data['name']
		artist = music_data['artist']
		y = music_data['y']
		sr = music_data['sr']
		beats = music_data['beats']
		bpm = music_data['bpm']
		frame_size = music_data['frame_size']
		melody = music_data['melody']
		acc = music_data['acc']
		section = music_data['section']
	"""
	"""
	def to_json(self):
		keys = ['FilePath', 'name', 'artist', 'genre', 'y', 'sr', 'beats', 'bpm',
			'frame_size','quantize', 'melody', 'acc', 'chords', 'section']
		values = [self.FilePath, self.name, self.artist, self.genre, self.y, 
			self.sr, self.beats, self.bpm, self.frame_size, self.quantize,
			self.melody, self.acc, self.chords, self.section]
		return json.dumps(dict(zip(keys, values)))
	"""
	"""
	def insert_db(self, cur):
		query = '''
INSERT INTO MUSIC(FilePath, Title, Artist, Genre, y, sr, beats, bpm, 
	frame_size, quantize, melody, acc, chords, section)
VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);
		'''
		data = (self.FilePath, self.name, self.artist, self.genre,
			self.y, self.sr, self.beats, self.bpm, self.frame_size, 
			self.quantize, self.melody, self.acc, self.chords, self.section)
		cur.execute(query, data)
	"""
	def load_database(self, data):
		(self.ID, self.FilePath, self.sr, self.beats, self.bpm, self.frame_size,
		self.quantize, self.acc, self.chords) = tuple(data)
		self.y, self.sr = librosa.load(self.FilePath, sr=None, mono=False)
		self.y = librosa.util.normalize(self.y, axis=1)
	def analyze_beats(self):
		self.bpm, self.beats = librosa.beat.beat_track(y=librosa.to_mono(self.y), sr=self.sr)
	def analyze_music(self):
		if self.predictions is None:
			self.predictions = spleeter_4stems_separate(self.y)         #spleeterによる音源分離
		vocals = self.predictions["vocals"]                             #ボーカル音源
		acc = self.predictions["other"]                                 #和音進行解析に使用する音源
		melody = f0_in_beats(self.y, self.sr, beats=self.beats, vocals=vocals, quantize=self.quantize*2)
			#ボーカルによる主旋律の解析をする
		self.melody = sep_count(melody)
		self.acc = chroma_in_beats(self.y, self.sr, self.beats, acc)    #拍ごとのクロマグラムを作成する
		self.chords = estimate_chords(self.acc)                         #和音進行を推定する
	def analyze(self, quantize=4):
		self.quantize = int(quantize)
		self.analyze_beats()
		self.sep_beats(self.quantize)
		self.analyze_music()
	def sep_beats(self, quantize):
		self.beats = sep_quantize(self.beats, quantize)
		self.bpm = self.bpm * quantize/4
