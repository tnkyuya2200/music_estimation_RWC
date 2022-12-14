#coding:utf-8
import numpy as np
import librosa
from tqdm import tqdm
# from spleeter.separator import Separator
from collections import OrderedDict
import sqlite3
import mysql.connector
import io
import copy
import os
import csv
import warnings
import json 
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
    input_melody_nan = none_to_nan(input_melody)
    database_melody_nan = none_to_nan(database_melody)
    
    for index_db in range(len(database_melody_nan)):
        db_sample = database_melody_nan[index_db]
        sim_db = [0]
        for index_input in range(len(input_melody_nan)):
            input_sample = input_melody_nan[index_input]
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
    return sim_melody, sim_chords

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
            output.append(None)
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

def nan_to_none(data):
    result = copy.deepcopy(data)
    for group in result:
        for idx, frame in enumerate(group):
            if np.isnan(frame):
                group[idx] = None
    return result

def none_to_nan(data):
    if data is None:
        return None
    result = copy.deepcopy(data)
    for group in result:
        for idx, frame in enumerate(group):
            if frame is None:
                group[idx] = np.nan
    return result


class Database:
    con = None
    cur = None
    def __init__(self):
        """
        input: None
        """
        self.con = mysql.connector.connect(
            user="root",
            host="localhost",
            database="python_db",
            unix_socket="/data/mysql/mysql.sock"
        )
        if not self.con.is_connected:
            return

        self.cur = self.con.cursor()
    def __del__(self):
        self.cur.close()
        self.con.close()
    def init_database(self, csv_Path):
        self.cur = self.con.cursor()
        self.cur.execute("""
        DROP TABLE IF EXISTS 
            info, features, y, beats, 
            e_vocals, e_acc, melody, chords;
        """)
        self.cur.execute("""
CREATE TABLE info(
    ID INT NOT NULL PRIMARY KEY,
    NO INT,
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
)
        """)
        self.cur.execute("""
CREATE TABLE features(
    ID INTEGER PRIMARY KEY,
    FilePath TEXT,
    sr INTEGER,
    bpm INTEGER,
    frame_size INTEGER,
    quantize INTEGER
)
        """)
        self.cur.execute("""
CREATE TABLE y(
    ID INTEGER,
    idx1 INTEGER,
    idx2 INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx1, idx2)
)
        """)
        self.cur.execute("""
CREATE TABLE beats(
    ID INTEGER ,
    idx INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx)

)
        """)
        self.cur.execute("""
CREATE TABLE e_vocals(
    ID INTEGER,
    idx1 INTEGER,
    idx2 INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx1, idx2)
)
        """)
        self.cur.execute("""
CREATE TABLE e_acc(
    ID INTEGER,
    idx1 INTEGER,
    idx2 INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx1, idx2)
)
        """)
        self.cur.execute("""
CREATE TABLE melody(
    ID INTEGER,
    idx1 INTEGER,
    idx2 INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx1, idx2)
)
        """)
        self.cur.execute("""
CREATE TABLE chords(
    ID INTEGER,
    idx1 INTEGER,
    idx2 INTEGER,
    data FLOAT,
    PRIMARY KEY(ID, idx1, idx2)
)
        """)
        self.cur.execute("INSERT INTO features (ID) VALUES (0)")
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
    FilePath
) 
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)

"""
        feature_query = """
INSERT INTO features (
    ID, 
    FilePath
) 
VALUES (%s, %s)
"""
        self.cur.executemany(info_query, info_rows)
        self.cur.executemany(feature_query, feature_rows)
        open_csv.close()
        self.con.commit()
        
    def load_Music_by_ID(self, ID=0):
        """
        input:
            ID int: select data from database by ID
        output Music: loaded music
        """
        music = Music()

        #load features
        query = """
SELECT ID, FilePath, sr, bpm, frame_size, quantize 
FROM features 
WHERE ID = %s
        """
        self.cur.execute(query, (ID,))
        (music.ID, music.FilePath, music.sr, music.bpm, 
            music.frame_size, music.quantize) = self.cur.fetchone()

        #load y
        music.y = self.load_data("y")
        music.beats = self.load_data("beats")
        music.e_vocals = self.load_data("e_vocals")
        music.e_acc = self.load_data("e_acc")
        music.chords = self.load_data("chords")
        music.melody = self.load_data("melody")
        query = """
SELECT data FROM beats WHERE ID = %s ORDER BY idx ASC;
        """
        music.beats = np.array(
            [x[0] for x in self.cur.fetchone()], dtype="object"
        )
        self.cur.close()
        return music

    def getdbsize(self):
        """
        output int: database size
        """
        self.cur = self.con.cursor()
        query = "SELECT count(*) FROM features"
        self.cur.execute(query)
        self.cur.close()
        return self.cur.fetchone()[0]
    def insert_db(self, music):
        """
        input:
            music Music: music data to insert
        """
        query = """
INSERT INTO features (
    ID,
    FilePath,
    sr,
    bpm,
    frame_size,
    quantize
)
VALUES(%s, %s, %s, %s, %s, %s)
        """
        self.cur.execute(
            query, 
            (music.ID, music.FilePath, music.sr, 
            music.bpm, music.frame_size, music.quantize)
        )
        self.con.commit()
        
    def getIDlist(self):
        self.cur = self.con.cursor()
        query = "SELECT ID FROM features"
        self.cur.execute(query)
        self.cur.close()
        return list(map(lambda x:x[0], self.cur.fetchall()))

    def loadAllMusic(self):
        IDlist = self.getIDlist()
        music_list = list(map(load_Music_by_ID, IDlist))
        return music_list

    def insert_data(self, table_name, ID, data):
        self.cur = self.con.cursor()
        if table_name == "beats":
            query = """
INSERT INTO %s (
    ID, idx, data
)
VALUES (%s, %s, %s)
            """
            self.cur.executemany(
                query, [("beats", idx, x) for idx, x in enumerate(data)]
            )
        else:
            query = """
INSERT INTO %s (
    ID, idx1, idx2, data
)
VALUES (%s, %s, %s, %s)
            """
            for idx1, row in enumerate(data):
                for idx2, value in enumerate(row):
                    data.append((table_name, idx1, idx2, value))
            self.cur.executemany(query, data)
        # self.con.commit()
        self.cur.close()

    def load_data(self, ID, table_name):
        self.cur = self.con.cursor()
        if table_name == "beats":
            query = """
SELECT data FROM beats WHERE ID = %s ORDER BY idx ASC
            """
            self.cur.execute(query, (ID,))
            data = np.array(
                [x[0] for x in self.cur.fetchone()], 
                dtype="object"
            )
        else:
            query = """
SELECT data FROM %s WHERE ID = %s AND idx1 = %s ORDER BY idx2 ASC
        """
            idx1 = 0
            self.cur.execute(query, (table_name, ID, idx1))
            tmp_data = [x[0] for x in self.cur.fetchall()]
            data = []
            while len(data) != 0:
                data.append(tmp_data)
                idx1 += 1
                self.cur.execute(query, (ID, idx1))
                tmp_data = [x[0] for x in self.cur.fetchall()]
            data = np.array(data, dtype="object")
        self.cur.close()
        return data
    def delete_data(self, table_name, ID):
        self.cur = self.con.cursor()
        query = """
DELETE FROM %s WHERE ID = %s
        """
        self.cur.execute(query, (ID,))
    def update_data(self, table_name, ID, data):
        self.delete_data(table_name, ID)
        self.insert_data(table_name, ID, data)
class Music:
    ID = None           # Song ID                               int
    y = None            # wav series                            np.ndarray, shape=(2,samples)
    FilePath = None     # FilePath to file                      str
    sr = 0              # samplerate                            int
    beats = None        # frames where beat is                  np.ndarray, shape=(frames,)
    bpm = 0             # beats per minute                      int
    frame_size = 512    # frame size (default 512 samples)      int
    quantize = 4        # how often you get melody notes        int
    e_vocals = None     # estimated vocals                      np.ndarray, shape=(2,samples)
    e_acc = None        # estimated acc                         np.ndarray, shape=(2,samples)
    melody = None       # analyzed cqt_bins in each beats       np.ndarray, shape=(len(beats),)
    chords = None       # analyzed chords                       np.ndarray, shape=(12,len(beats))

    def load_music(self, FilePath):
        self.ID = 0
        self.FilePath = FilePath
    def load_and_analyze_music(self, quantize=4):
        self.y, self.sr = librosa.load(self.FilePath, mono=False)
        self.y = librosa.util.normalize(self.y, axis=1)
        self.analyze_music(quantize)
    def analyze_music(self, quantize=4):
        self.quantize = int(quantize)
        self.bpm, self.beats = librosa.beat.beat_track(y=librosa.to_mono(self.y), sr=self.sr)
        vocals_f0 = f0_in_beats(self.e_vocals, self.beats, self.sr)
        self.sep_beats(self.quantize)
        self.melody = sep_count(vocals_f0)
        self.chords = estimate_chords(chroma_in_beats(self.e_acc, self.sr, self.beats))
    # def separate_music(self):
    #     self.e_vocals, self.e_acc = spleeter_4stems_separate(self.y)
    def sep_beats(self, quantize):
        self.beats = sep_quantize(self.beats, quantize)
        self.bpm = self.bpm * quantize/4
    def to_list(self):
        return (self.ID, sql_conv_w(self.y), self.FilePath, self.sr, sql_conv_w(self.beats), self.bpm, self.frame_size,
            self.quantize, sql_conv_w(self.e_vocals), sql_conv_w(self.e_acc), sql_conv_w(self.melody), sql_conv_w(self.chords))
    def schema(self):
        return "ID, y, FilePath, sr, beats, bpm, frame_size, quantize, e_vocals, e_acc, melody, chords"
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