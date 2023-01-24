#coding:utf-8
import numpy as np
import librosa
from tqdm import tqdm
# from spleeter.separator import Separator
from collections import OrderedDict
import sqlite3
import io
import copy
import os
import csv
import warnings

chord_dic = [[1,0,0,0,1,0,0,1,0,0,0,0],   #C
[0,1,0,0,0,1,0,0,1,0,0,0],   #Db
[0,0,1,0,0,0,1,0,0,1,0,0],   #D
[0,0,0,1,0,0,0,1,0,0,1,0],   #Eb
[0,0,0,0,1,0,0,0,1,0,0,1],   #E
[1,0,0,0,0,1,0,0,0,1,0,0],   #F
[0,1,0,0,0,0,1,0,0,0,1,0],   #Gb
[0,0,1,0,0,0,0,1,0,0,0,1],   #G
[1,0,0,1,0,0,0,0,1,0,0,0],   #Ab
[0,1,0,0,1,0,0,0,0,1,0,0],   #A
[0,0,1,0,0,1,0,0,0,0,1,0],   #Bb
[0,0,0,1,0,0,1,0,0,0,0,1],   #B
#minor chords
[1,0,0,1,0,0,0,1,0,0,0,0],   #Cm
[0,1,0,0,1,0,0,0,1,0,0,0],   #Dbm
[0,0,1,0,0,1,0,0,0,1,0,0],   #Dm
[0,0,0,1,0,0,1,0,0,0,1,0],   #Ebm
[0,0,0,0,1,0,0,1,0,0,0,1],   #Em
[1,0,0,0,0,1,0,0,1,0,0,0],   #Fm
[0,1,0,0,0,0,1,0,0,1,0,0],   #Gbm
[0,0,1,0,0,0,0,1,0,0,1,0],   #Gm
[0,0,0,1,0,0,0,0,1,0,0,1],   #Abm
[1,0,0,0,1,0,0,0,0,1,0,0],   #Am
[0,1,0,0,0,1,0,0,0,0,1,0],   #Bbm
[0,0,1,0,0,0,1,0,0,0,0,1]]   #Bm

def pre_cos_sim(chord_dic = chord_dic):
    result = np.empty((24, 24))
    deno = np.linalg.norm(chord_dic[0]) * np.linalg.norm(chord_dic[0])
    for i, datai in enumerate(chord_dic):
        for j, dataj in enumerate(chord_dic):
            result[i][j] = np.dot(datai, dataj) / 3
    return result

sim_chords = pre_cos_sim()

def estimate_chord(chroma, chord_dic=chord_dic):
    """
    input:
        chroma np.ndarray, shape(12,): chroma vector
        chord_dic dict: chord dictionary
    output int: most similar chord idx
    """
    maximum = -1
    this_chord = 0
    for chord_index, vector in enumerate(chord_dic):
        similarity = cos_sim(chroma, vector)
        if similarity > maximum:
            maximum = similarity
            this_chord = chord_index
    return this_chord

def estimate_chords(chromas, chord_dic=chord_dic):
    """
    input:
        output np.ndarray, shape=(12,len(beats)): analyzed chroma in beats
        chord_dic dict: chord dictionary
    output np.ndarray, shape=(len(beats)): most similar chords in beats
    """
    result = np.empty(chromas.shape[1], dtype=np.int8)
    for i in range(chromas.shape[1]):
        result[i] = estimate_chord(chromas[:,i], chord_dic)
    return result

def chroma_in_beats(acc_wav, sr, beats):
    """
    input:
        acc_wav np.ndarray, shape=(2,samples): acc wave series
        sr int: sampling rate
        beats np.ndarray, shape=(frames,): analyzed beats
    output np.ndarray, shape=(12,len(beats)): analyzed chroma in beats
    """
    TONES = 12
    chroma = librosa.feature.chroma_cens(librosa.to_mono(acc_wav), sr)
    sum_chroma = np.zeros((TONES, len(beats)+1))
    beats_count = 0
    for frame_index in range(chroma.shape[1]):
        for i in range(TONES):
            sum_chroma[i, beats_count] += np.abs(chroma[i, frame_index])
        if frame_index in beats:
            beats_count += 1
    return sum_chroma

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

def sep_by_nan(tmp_melody, threshold=1):
    """
    input:
        tmp_melody np.ndarray, shape=(len(beats),): non-separated analyzed melody
        threshold int: vocals are separated by continuous nan value = threshold
    output: analyzed melody separated by nan:  np.ndarray, shape=(threshold, array)
    separates vocal by thresholded nan
    """
    result = []
    cont = []
    nan_count = 0
    for d in tmp_melody:
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

def sep_count(vocals_f0, count=10, threshold=4):
    """
    input:
        vocals_f0 np.ndarray, shape=(len(beats*2)): analyzed f0 of vocals in beats
        count int: vocals are separated minimum count > count
        threshold int: default threshold for sep_by_nan
    output np.ndarray, shape=(count, array): analyzed melody seprated > count
    """
    result = sep_by_nan(vocals_f0, threshold)
    while len(result) > count:
        threshold += 1
        result = sep_by_nan(vocals_f0, threshold)
    result = np.array(result, dtype=object)
    return result

def compare_melody(input_melody, database_melody):
    """
    input: compare 2 melody
        input_melody: melody1 to compare: np.ndarray, shape=(count, array)
        database_melody: melody2 to compare: np.ndarray, shape=(count, array)
    output float64: simirality score
    """
    #sim = []
    sim = np.empty(len(database_melody))
    for index_db in range(len(database_melody)):
        db_sample = database_melody[index_db]
        #sim_db = [0]
        sim_db = np.empty(len(input_melody)+1)
        sim_db[-1] = 0
        for index_input in range(len(input_melody)):
            input_sample = input_melody[index_input]
            #sim_lag = [0]
            sim_lag = np.empty(abs(len(input_sample)-len(db_sample))+2)
            sim_lag[-1] = 0
            if len(input_sample) > len(db_sample):
                for lag in range(len(input_sample)-len(db_sample)+1):
                    #diff = []
                    diff = np.empty(len(db_sample))
                    both_nan = 0
                    for i in range(len(db_sample)):
                        #diff.append((input_sample[i+lag] - db_sample[i]))
                        diff[i] = input_sample[i+lag] - db_sample[i]
                        if np.isnan(input_sample[i+lag]) and np.isnan(db_sample[i]):
                            both_nan += 1
                    if not all(np.isnan(diff)):
                        med = np.median([x for x in diff if not np.isnan(x)])
                        #sim_lag.append((len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff))
                        sim_lag[lag] = (len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff)
                    else:
                        #sim_lag.append(0)
                        sim_lag[lag] = 0
            else:
                for lag in range(len(db_sample)-len(input_sample)+1):
                    #diff = []
                    diff = np.empty(len(input_sample))
                    both_nan = 0
                    for i in range(len(input_sample)):
                        #diff.append((input_sample[i] - db_sample[i+lag]))
                        diff[i] = input_sample[i] - db_sample[i+lag]
                        if np.isnan(input_sample[i]) and np.isnan(db_sample[i+lag]):
                            both_nan += 1
                    if not all(np.isnan(diff)):
                        med = np.median([x for x in diff if not np.isnan(x)])
                        #sim_lag.append((len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff))
                        sim_lag[lag] = (len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff)
                    else:
                        #sim_lag.append(0)
                        sim_lag[lag] = 0
            #sim_db.append(max(sim_lag))
            sim_db[index_input] = max(sim_lag)
        #sim.append(max(sim_db))
        sim[index_db] = max(sim_db)
    return np.mean(sim)

def cos_sim(v1,v2):
    """
    input:
        v1 array: vector
        v2 array: vector
    output float64: cosine simirality of v1, v2
    """
    #return sim_chords[v1, v2]
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def corr_cossim(data1, data2):
    """
    input:
        data1 np.ndarray: vector
        data2 np.ndarray: vector
    output float64: cosine simirality for different size vector
    calcurates mean vector cosine simirality
    """
    length = min(data1.shape[0], data2.shape[0])
    if length == 0:
        return 0
    #sim_i = []
    #sim_i = np.empty(min(data1.shape[0], data2.shape[0]))
    result = 0
    for i in range(length):
        #sim_i.append(cos_sim(data1[i], data2[i]))
        #sim_i[i] = sim_chords[data1[i], data2[i]]
        result += sim_chords[data1[i], data2[i]]
    return result/length
    #return np.mean(sim_i)

def roll_chords(acc, i):
    """
    result = np.empty(acc.shape, dtype="int8")
    for idx, data in enumerate(acc):
        if data < 12:
            result[idx] = (data+i)%12
        else:
            result[idx] = (data-12+i)%12+12
    """
    return np.array([(data+i)%12 if data<12 else (data-12+i)%12+12 for data in acc])

def compare_acc(acc1, acc2, separate=64):
    """
    input:
        acc1 np.ndarray, shape=(len(beats)): input acc
        acc2 np.ndarray, shape=(len(beats)): database acc
        separate int: separate shorter acc by separate
    output float64: simirality of acc1 and acc2
    """
    shorter_acc = None
    longer_acc = None
    if acc1.shape[0] < acc2.shape[0]:
        shorter_acc = acc2
        longer_acc = acc1
    else:
        shorter_acc = acc1
        longer_acc = acc2
    #sim_i = []
    sim_i = np.empty(12)
    for i in range(12):
        rolled_shorter_acc = roll_chords(shorter_acc, i)
        #sim = []
        sim = np.empty(shorter_acc.shape[0]//separate)
        for index_shorter in range(shorter_acc.shape[0]//separate):
            shorter_sample = rolled_shorter_acc[index_shorter*separate:min((index_shorter+1)*separate, shorter_acc.shape[0]-1)]
            #sim_index = [0]
            sim_index = np.empty(longer_acc.shape[0]-separate+1)
            sim_index[-1] = 0
            for index_longer in range(longer_acc.shape[0]-separate):
                longer_sample = longer_acc[index_longer:index_longer+separate]
                #sim_index.append(corr_cossim(shorter_sample, longer_sample))
                sim_index[index_longer] = corr_cossim(shorter_sample, longer_sample)
            #sim.append(max(sim_index))
            sim[index_shorter] = max(sim_index)
        #sim_i.append(np.mean(sim))
        sim_i[i] = np.mean(sim)
    return max(sim_i)

def compare(input_music, database_music):
    """
    input:
        input_music Music: music to compare
        database_music Music: music to compare
    output (sim_melody, sim_chords):
        sim_melody float64: simirality of melody
        sim_chords float64: simirality of acc
    """
    sim_melody = compare_melody(input_music.melody, database_music.melody)
    #sim_acc = compare_acc(input.acc, data.acc)
    sim_chords = compare_acc(input_music.chords, database_music.chords)
    return sim_melody,  sim_chords

# def spleeter_4stems_separate(y, sep=20000000):
#     """
#     input:
#         y np.ndarray, shape=(2,samples): input wav series
#         sep int: separate unit
#     output (vocals, other):
#         vocals np.ndarray, shape=(2,samples): predicted vocals
#         other np.ndarray, shape=(2,samples): predicted acc
#     """
#     vocals = np.empty(y.shape)
#     other = np.empty(y.shape)
#     separator = Separator("spleeter:4stems")
#     for i in range(0, y.shape[1], sep):
#         prediction = separator.separate(y[:, i:min(i+sep, y.shape[1])].T)
#         vocals[:, i:min(i+sep, y.shape[1])] = prediction["vocals"].T
#         other[:, i:min(i+sep, y.shape[1])] = prediction["other"].T
#     return (vocals, other)

def sep_quantize(arr, quantize):
    """
    input:
        arr array: beat flames array
        quantize int: quantize
    output array: quantized beat frames array
    """
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

def f0_in_beats(vocals, beats, sr, quantize=8):
    """
    input:
        vocals np.ndarray, shape=(2,samples): estimated vocals
        beats np.ndarray, shape=(frames,): analyzed beats
        sr int: sampling rate
        quantize int: how often you separate beats
    output np.ndarray shape=(quantized_beats_frames,): analyzed f0 in each quantized beats
    """
    q_beats = sep_quantize(beats, 8)
    f0 = (list(map(lambda x:round(x-12), librosa.hz_to_midi(librosa.yin(librosa.to_mono(vocals),fmin=65,fmax=2093,sr=sr)))))
    #output = []
    output = np.empty(len(q_beats)-1)
    threshold = np.percentile(abs(vocals), 25)
    mono_vocals = librosa.to_mono(vocals)
    for index in range(len(q_beats)-1):
        note = np.median(f0[q_beats[index]:q_beats[index+1]])
        if np.median(np.abs(mono_vocals[q_beats[index]*512:q_beats[index+1]*512])) < threshold or note >= 80 :
            #output.append(np.nan)
            output[index] = np.nan
        else:
            #output.append(note)
            output[index] = note
    return output
    #return np.array(output)

def compare_all(test_music, db):
    """
    input:
        test_music Music: Music object to compare
        db str: Database object to compare
    output dict{"sim":"vocals", "chords", "average"}:
        "sim":"vocals" float64: vocal simirality score
        "sim":"chords" float64: chords simirality score
        "sim":"average" float64: final simirality score
    """
    test_music.analyze_music(4)

    result = {}

    for ID in tqdm(range(1,db.getdbsize()), leave=False, position=1):
        result[ID] = {"sim":{}}
        x = db.load_Music_by_ID(ID)
        if test_music.bpm < x.bpm*3/4:
            x_q2 = copy.deepcopy(x)
            x_q2.analyze_music(2)
            vocal_sim, chords_sim = compare(test_music, x_q2)
            result[ID]["sim"]["vocal"] = vocal_sim
            result[ID]["sim"]["chords"] = chords_sim
            result[ID]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
        elif test_music.bpm > x.bpm*3/2:
            test_q2 = copy.deepcopy(test_music)
            test_q2.analyze_music(2)
            vocal_sim, chords_sim = compare(test_q2, x)
            result[ID]["sim"]["vocal"] = vocal_sim
            result[ID]["sim"]["chords"] = chords_sim
            result[ID]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
        else:
            vocal_sim, chords_sim = compare(test_music, x)
            result[ID]["sim"]["vocal"] = vocal_sim
            result[ID]["sim"]["chords"] = chords_sim
            result[ID]["sim"]["average"] = np.mean((vocal_sim, chords_sim))
    return result

class Database:
    Path = None
    con = None
    cur = None
    def __init__(self, Path="music.db"):
        """
        input:
            Path str: FilePath to database
        """
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        self.Path = Path
        self.con = sqlite3.connect(Path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        self.con.isolation_level = None
    def __del__(self):
        self.con.close()
    def init_database(self, csv_Path="Data_light.csv"):
        self.con.execute("DROP TABLE IF EXISTS info;")
        self.con.execute("DROP TABLE IF EXISTS features;")
        info_query = """
CREATE TABLE IF NOT EXISTS info(
    ID INTEGER PRIMARY KEY,
    NO INTEGER,
    Composer TEXT,
    Composer_Eng TEXT,
    Artist TEXT,
    Artist_Eng TEXT,
    Title TEXT,
    Title_Eng TEXT,
    CD TEXT,
    Track_No TEXT,
    Genre TEXT,
    Genre_Eng TEXT,
    Sub_Genre TEXT,
    Sub_Genre_ENG TEXT,
    FilePath TEXT
);
        """
        feature_query = """
CREATE TABLE IF NOT EXISTS features(
    ID INTEGER PRIMARY KEY,
    FilePath TEXT,
    y ARRAY,
    sr INTEGER,
    beats ARRAY,
    bpm INTEGER,
    frame_size INTEGER,
    quantize INTEGER,
    esti_vocals ARRAY,
    esti_acc ARRAY,
    melody ARRAY,
    chords ARRAY
);
        """
        self.con.execute(info_query)
        self.con.execute(feature_query)
        self.con.execute("INSERT INTO features (ID) VALUES (0);")
        open_csv = open(csv_Path, encoding="utf-8")
        read_csv = csv.reader(open_csv)

        info_rows = []
        feature_rows = []
        for row in read_csv:
            info_rows.append(row)
            feature_rows.append([row[0], row[14]])
        info_query = """
INSERT INTO info (
ID,
NO,
Composer,
Composer_Eng,
Artist,
Artist_Eng,
Title,
Title_Eng,
CD,
Track_No,
Genre,
Genre_Eng,
Sub_Genre,
Sub_Genre_ENG,
FilePath)
VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""
        feature_query = """
INSERT INTO features (
    ID, 
    FilePath
) 
VALUES (?, ?)
"""

        self.cur.executemany(info_query, info_rows)
        self.cur.executemany(feature_query, feature_rows)
        open_csv.close()

    def load_Music_by_ID(self, ID=0):
        """
        input:
            ID int: select data from database by ID
        output Music: loaded music
        """
        music = Music()
        query = "select ID, y, FilePath, sr, beats, bpm, frame_size, quantize, esti_vocals, esti_acc, melody, chords from features where ID = ?;"
        self.cur.execute(query, (ID,))
        music.load_database(self.cur.fetchone())
        return music
    def getdbsize(self):
        """
        output int: database size
        """
        query = "select count(*) from features;"
        self.cur.execute(query)
        return self.cur.fetchone()[0]
    def insert_db(self, music):
        """
        input:
            music Music: music data to insert
        """
        query = "INSERT INTO features ("+ music.schema() +") VALUES(?,?,?,?,?,?,?,?,?,?,?,?);"
        self.cur.execute(query, music.to_list())
    def getIDlist(self):
        query = "select ID from features;"
        self.cur.execute(query)
        return list(map(lambda x:x[0], self.cur.fetchall()))
    def loadAllMusic(self):
        query = "select ID, y, FilePath, sr, beats, bpm, frame_size, quantize, esti_vocals, esti_acc, melody, chords from features;"
        self.cur.execute(query)
        output_list = self.cur.fetchall()
        music_list = []
        for output in output_list:
            music = Music()
            music.load_database(output)
            music_list.append(music)
        return music_list
        
class Music:
    ID = None           # Song ID                               int
    y = None            # wav series                            np.ndarray, shape=(2,samples)
    FilePath = None     # FilePath to file                      str
    sr = 0              # samplerate                            int
    beats = None        # frames where beat is                  np.ndarray, shape=(frames,)
    bpm = 0             # beats per minute                      int
    frame_size = 512    # frame size (default 512 samples)      int
    quantize = 4        # how often you get melody notes        int
    esti_vocals = None  # estimated vocals                      np.ndarray, shape=(2,samples)
    esti_acc = None     # estimated acc                         np.ndarray, shape=(2,samples)
    melody = None       # analyzed cqt_bins in each beats       np.ndarray, shape=(len(beats),)
    chords = None       # analyzed chords                       np.ndarray, shape=(len(beats))

    def load_music(self, FilePath):
        self.ID = 0
        self.FilePath = FilePath
    def load_database(self, data):
        (self.ID, self.y, self.FilePath, self.sr, self.beats, self.bpm, self.frame_size, self.quantize,
        self.esti_vocals, self.esti_acc, self.melody, self.chords) = tuple(data)
    def load_and_analyze_music(self, quantize=4):
        self.y, self.sr = librosa.load(self.FilePath, mono=False)
        self.y = librosa.util.normalize(self.y, axis=1)
        self.analyze_music(quantize)
    def analyze_music(self, quantize=4):
        self.quantize = int(quantize)
        self.bpm, self.beats = librosa.beat.beat_track(y=librosa.to_mono(self.y), sr=self.sr)
        vocals_f0 = f0_in_beats(self.esti_vocals, self.beats, self.sr)
        self.sep_beats(self.quantize)
        self.melody = sep_count(vocals_f0)
        self.chords = estimate_chords(chroma_in_beats(self.esti_acc, self.sr, self.beats))
    # def separate_music(self):
    #     self.esti_vocals, self.esti_acc = spleeter_4stems_separate(self.y)
    def sep_beats(self, quantize):
        self.beats = sep_quantize(self.beats, quantize)
        self.bpm = self.bpm * quantize/4
    def to_list(self):
        return (self.ID, self.y, self.FilePath, self.sr, self.beats, self.bpm, self.frame_size,
            self.quantize, self.esti_vocals, self.esti_acc, self.melody, self.chords)
    def schema(self):
        return "ID, y, FilePath, sr, beats, bpm, frame_size, quantize, esti_vocals, esti_acc, melody, chords"
    def cqt_AF(self):
        y_cqt = librosa.cqt(librosa.to_mono(m.y))
        frame_result = np.ndarray((y_cqt.shape[0]-1, y_cqt.shape[1]), dtype='bool')
        result = np.ndarray((y_cqt.shape[1]), dtype='int32')
        for frame in range(y_cqt.shape[1]-1):
            for nbin in range(y_cqt.shape[0]-1):
                frame_result[nbin, frame] = \
                    y_cqt[nbin][frame+1] - y_cqt[nbin+1][frame+1] -\
                        y_cqt[nbin][frame] + y_cqt[nbin+1][frame] > 0
                result.append(frame_result)
        return result
    def cqt_beat_AF(self):
        y_cqt = librosa.cqt(librosa.to_mono(m.y))
        frame_result = np.ndarray((y_cqt.shape[0]-1, y_cqt.shape[1]), dtype='bool')
        result = np.ndarray((y_cqt.shape[1]), dtype='int32')
        for n_beat in range(len(self.beats)-2):
            for n_bin in range(y_cqt.shape[0]-1):
                beat_result[n_bin, n_beat] = \
                    conv_stft(n_bin, n_beat+1) - conv_stft(n_bin+1, n_beat+1) -\
                        conv_stft(n_bin, n_beat) + conv_stft(n_bin+1, n_beat) > 0
                result.append(beat_result)
        return result
