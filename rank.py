#coding:utf-8
import myfunctions as fn
import numpy as np
import sys, os
from datetime import datetime
from tqdm import tqdm
import json

def get_db(result_path):
    db = fn.Database()
    query = "select name, artist from music where path == ?;"
    db.cur.execute(query, (result_path,))
    name, artist = db.cur.fetchall()[0]
    result = name + " by " + artist
    return result

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

file = open(sys.argv[1], "r", encoding="utf-8")
input_dict = json.load(file)

for input_filename, db_files in tqdm(input_dict.items()):
    if not(type(db_files) is dict):
        continue
    sim_rank_in_vocal = sorted(db_files, key=lambda x:db_files[x]["sim"]["vocal"], reverse=True)
    sim_rank_in_chords = sorted(db_files, key=lambda x:db_files[x]["sim"]["chords"], reverse=True)
    sim_rank_in_all = sorted(db_files, key=lambda x:(db_files[x]["sim"]["vocal"]+db_files[x]["sim"]["chords"])/2, reverse=True)
    #sim_rank_in_all = sorted(db_files, key=lambda x:db_files[x]["sim"]["average"], reverse=True)

    print(input_filename + " is estimated to be ")
    print("\tin vocal")
    for i in range(3):
        print("\t\t"+get_db(sim_rank_in_vocal[i])+"\n\t\t\tscore: "+str(db_files[sim_rank_in_vocal[i]]["sim"]["vocal"]))
    print("\tin chords")
    for i in range(3):
        print("\t\t"+get_db(sim_rank_in_chords[i])+"\n\t\t\tscore: "+str(db_files[sim_rank_in_chords[i]]["sim"]["chords"]))
    print("\tin all features")
    for i in range(3):
        print("\t\t"+get_db(sim_rank_in_all[i])+"\n\t\t\tscore: "+str((db_files[sim_rank_in_all[i]]["sim"]["vocal"]+db_files[sim_rank_in_all[i]]["sim"]["chords"])/2))
        #print("\t\t"+get_db(sim_rank_in_all[i])+"\n\t\t\tscore: "+str(db_files[sim_rank_in_all[i]]["sim"]["average"]))
    print("\n")
    sim_rank_in_all[0]["ID"]