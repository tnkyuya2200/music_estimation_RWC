#coding:utf-8
import sqlite3
import json
import myfunctions as fn
import sys, os
import json
from datetime import datetime
import numpy as np
import warnings
from tqdm import tqdm
warnings.simplefilter("error")

db = fn.Database(sys.argv[1])
IDs = db.getIDlist()

result = {"test_file": sys.argv[2]}
result["db"] = [] 
filename = os.path.join(sys.argv[3], os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json")
test_music = db.load_Music_by_ID(0)
test_music.analyze_music(4)
test_music_q2 = db.load_Music_by_ID(0)
test_music_q2.analyze_music(2)

for ID in tqdm(IDs[1:], desc="[estimating "+sys.argv[2]+"]"):
	tmp_dict = {"ID":ID, "sim":{}}
	x = db.load_Music_by_ID(ID)
	vocal_sim, chords_sim = (0, 0)
	if test_music.bpm < x.bpm*3/4:
		x_q2 = db.load_Music_by_ID(ID)
		x_q2.analyze_music(2)
		vocal_sim, chords_sim = fn.compare(test_music, x_q2)
	elif test_music.bpm > x.bpm*3/2:
		vocal_sim, chords_sim = fn.compare(test_music_q2, x)
	else:
		vocal_sim, chords_sim = fn.compare(test_music, x)
	tmp_dict["sim"]["vocal"] = vocal_sim
	tmp_dict["sim"]["chords"] = chords_sim
	tmp_dict["sim"]["average"] = np.mean((vocal_sim, chords_sim))
	result["db"].append(tmp_dict)

result["timestamp"] = datetime.now().isoformat()

file = open(filename, "w", encoding="utf-8")
json.dump(result, file, indent=2, ensure_ascii=False)
file.close()
print("written " + filename)

#os.system("shutdown -s -t 60")
