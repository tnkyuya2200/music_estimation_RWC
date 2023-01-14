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
from bitarray import bitarray

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
    """
    input:
        chroma np.ndarray, shape(12,): chroma vector
        chord_dic dict: chord dictionary
    output np.ndarray, shape=(12,): most similar chord vector
    """
    maximum = -1
    this_chord = np.zeros(12)
    for chord_index, (name, vector) in enumerate(chord_dic.items()):
        similarity = cos_sim(chroma, vector)
        if similarity > maximum:
            maximum = similarity
            this_chord = vector
    return this_chord

def estimate_chords(chromas, chord_dic=chord_dic):
    """
    input:
        output np.ndarray, shape=(12,len(beats)): analyzed chroma in beats
        chord_dic dict: chord dictionary
    output np.ndarray, shape=(12, len(beats)): most similar chords in beats
    """
    result = np.empty(chromas.shape)
    for i in range(chromas.shape[1]):
        result[:,i] = estimate_chord(chromas[:,i], chord_dic)
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
                    if not all(np.isnan(diff)):
                        med = np.median([x for x in diff if not np.isnan(x)])
                        sim_lag.append((len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff))
                    else:
                        sim_lag.append(0)
            else:
                for lag in range(len(db_sample)-len(input_sample)+1):
                    diff = []
                    both_nan = 0
                    for i in range(len(input_sample)):
                        diff.append((input_sample[i] - db_sample[i+lag]))
                        if np.isnan(input_sample[i]) and np.isnan(db_sample[i+lag]):
                            both_nan += 1
                    if not all(np.isnan(diff)):
                        med = np.median([x for x in diff if not np.isnan(x)])
                        sim_lag.append((len([x for x in diff if abs(x-med)<=0.6])+both_nan) / len(diff))
                    else:
                        sim_lag.append(0)
            sim_db.append(max(sim_lag))
        sim.append(max(sim_db))
    return np.mean(sim)

def cos_sim(v1,v2):
    """
    input:
        v1 array: vector
        v2 array: vector
    output float64: cosine simirality of v1, v2
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def corr_cossim(data1, data2):
    """
    input:
        data1 np.ndarray: vector
        data2 np.ndarray: vector
    output float64: cosine simirality for different size vector
    calcurates mean vector cosine simirality
    """
    sim_i = []
    for i in range(min(data1.shape[1], data2.shape[1])):
        sim_i.append(cos_sim(data1[:,i], data2[:,i]))
    return np.mean(sim_i)

def compare_acc(acc1, acc2, separate=64):
    """
    input:
        acc1 np.ndarray, shape=(12,len(beats)): input acc
        acc2 np.ndarray, shape=(12,len(beats)): database acc
        separate int: separate shorter acc by separate
    output float64: simirality of acc1 and acc2
    """
    shorter_acc = None
    longer_acc = None
    if acc1.shape[1] < acc2.shape[1]:
        shorter_acc = acc2
        longer_acc = acc1
    else:
        shorter_acc = acc1
        longer_acc = acc2
    sim_i = []
    for i in range(12):
        rolled_shorter_acc = np.roll(shorter_acc, i, axis=0)
        sim = []
        for index_shorter in range(shorter_acc.shape[1]//separate):
            shorter_sample = rolled_shorter_acc[:,index_shorter*separate:min((index_shorter+1)*separate, shorter_acc.shape[1]-1)]
            sim_index = [0]
            for index_longer in range(longer_acc.shape[1]-separate):
                longer_sample = longer_acc[:,index_longer:index_longer+separate]
                sim_index.append(corr_cossim(shorter_sample, longer_sample))
            sim.append(max(sim_index))
        sim_i.append(np.mean(sim))
    return max(sim_i)

def compare_fp(fp1, fp2):
    """
    input:
        fp1 np.ndarray, shape=(len(beats), 83(nbin))
        fp2 np.ndarray, shape=(len(beats), 83(nbin))
    output float64: simirality of fp1, fp2
    """

    def BER(s1, s2):
        result = 0
        for i in range(s1.shape[0]):
            for j in range(s1.shape[1]):
                result += s1[i,j] == s2[i,j]
        return result

    separate = 64
    shorter_fp = fp1
    longer_fp = fp2
    if fp1.shape[0] > fp2.shape[0]:
        shorter_fp = fp2
        longer_fp = fp1
    sim_i = []
    for i in range(12):
        rolled_shorter_fp = shorter_fp[:, i:-(12-i)]
        sim_idx = [0]
        for shorter_idx in range(shorter_fp.shape[0]//separate):
            shorter_sample = rolled_shorter_fp[shorter_idx*separate:min((shorter_idx+1)*separate, shorter_fp.shape[1]-1), :]
            for longer_idx in range(longer_acc.shape[0]-separate):
                longer_sample = longer_acc[longer_idx:longer_idx+separate, :]
                sim_idx.append(BER(shorter_sample, longer_sample))
            sim.append(max(sim_idx))
        sim_i.append(np.mean(sim))
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
    output = []
    threshold = np.percentile(abs(vocals), 25)
    mono_vocals = librosa.to_mono(vocals)
    for index in range(len(q_beats)-1):
        note = np.median(f0[q_beats[index]:q_beats[index+1]])
        if np.median(np.abs(mono_vocals[q_beats[index]*512:q_beats[index+1]*512])) < threshold or note >= 80 :
            output.append(np.nan)
        else:
            output.append(note)
    return np.array(output)

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
    chords ARRAY,
    fingerprint ARRAY
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
        query = "select " + music.schema() + " from features where ID = ?;"
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
        query = "INSERT INTO features ("+ music.schema() +") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?);"
        self.cur.execute(query, music.to_list())
    def getIDlist(self):
        query = "select ID from features;"
        self.cur.execute(query)
        return list(map(lambda x:x[0], self.cur.fetchall()))
    def loadAllMusic(self):
        query = "select " + Music.schema() + " from features;"
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
    chords = None       # analyzed chords                       np.ndarray, shape=(12, len(beats))
    fingerprint = None  # fingerprint                           np.ndarray, shape=(len(beats), 83(nbin))

    def load_music(self, FilePath):
        self.ID = 0
        self.FilePath = FilePath
    def load_database(self, data):
        (self.ID, self.y, self.FilePath, self.sr, self.beats, self.bpm, self.frame_size, self.quantize,
        self.esti_vocals, self.esti_acc, self.melody, self.chords, self.fingerprint) = tuple(data)
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
        self.fingerprint = self.cqt_beat_AF()
    # def separate_music(self):
    #     self.esti_vocals, self.esti_acc = spleeter_4stems_separate(self.y)
    def sep_beats(self, quantize):
        self.beats = sep_quantize(self.beats, quantize)
        self.bpm = self.bpm * quantize/4
    def to_list(self):
        return (self.ID, self.y, self.FilePath, self.sr, self.beats, self.bpm, self.frame_size,
            self.quantize, self.esti_vocals, self.esti_acc, self.melody, self.chords, self.fingerprint)
    def schema(self):
        return "ID, y, FilePath, sr, beats, bpm, frame_size, quantize, esti_vocals, esti_acc, melody, chords, fingerprint"
    def cqt_beat_AF(self):
        nbin_beat_sum = lambda nbin, beat: np.sum(y_cqt[nbin][beat:beat+1])
        y_cqt = librosa.cqt(librosa.to_mono(self.y))
        result = np.ndarray((len(m.beats[:-2])), dtype='object')
        for beat_idx, beat in enumerate(m.beats[:-2]):
            beat_result = bitarray(y_cqt.shape[0]-1)
            for nbin in range(y_cqt.shape[0]-1):
                beat_result[nbin] = \
                    nbin_beat_sum(nbin+1, beat) + nbin_beat_sum(nbin, beat+1) \
                        - nbin_beat_sum(nbin+1, beat+1) - nbin_beat_sum(nbin, beat) > 0
        return result