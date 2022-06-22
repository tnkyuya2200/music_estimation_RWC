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
import joblib
from pathos.multiprocessing import ProcessingPool
from typing import Optional
import concurrent.futures
import copy
warnings.simplefilter("error")

def task(test_music, test_music_q2, x, ID):
	tmp_dict = {"ID":ID, "sim":{}}
	#x = db.load_Music_by_ID(ID)
	vocal_sim, chords_sim = (0, 0)
	if test_music.bpm < x.bpm*3/4:
		#x_q2 = db.load_Music_by_ID(ID)
		x_q2 = copy.deepcopy(x)
		x_q2.analyze_music(2)
		vocal_sim, chords_sim = fn.compare(test_music, x_q2)
	elif test_music.bpm > x.bpm*3/2:
		vocal_sim, chords_sim = fn.compare(test_music_q2, x)
	else:
		vocal_sim, chords_sim = fn.compare(test_music, x)
	tmp_dict["sim"]["vocal"] = vocal_sim
	tmp_dict["sim"]["chords"] = chords_sim
	tmp_dict["sim"]["average"] = np.mean((vocal_sim, chords_sim))
	return tmp_dict

def main():
	db = fn.Database(sys.argv[1])
	IDs = db.getIDlist()

	result = {"test_file": sys.argv[2]}
	result["db"] = [] 
	filename = os.path.join(sys.argv[3], os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json")
	test_music = db.load_Music_by_ID(0)
	test_music.load_and_analyze_music(4)
	test_music_q2 = copy.deepcopy(test_music)
	test_music_q2.analyze_music(2)
	
	#x_q2_list = []
	#result["db"] = joblib.Parallel(n_jobs=-1, verbose=2)(joblib.delayed(task)(test_music, test_music_q2, x, x_q2, ID) for test_music, test_music_q2, x, x_q2, ID in zip([test_music]*len(IDs), [test_music_q2]*len(IDs), x_list, x_q2_list, IDs[1:]))

	with concurrent.futures.ProcessPoolExecutor() as executor:
		with tqdm(total=len(IDs[1:]), desc="estimating "+sys.argv[2]) as progress:
			futures = []
			result["db"] = []
			for ID in IDs[1:]:
				future = executor.submit(
					task, test_music, test_music_q2,
					db.load_Music_by_ID(ID), ID
					)
				future.add_done_callback(lambda p: progress.update())
				futures.append(future)
		result["db"] = [f.result() for f in futures]
	result["timestamp"] = datetime.now().isoformat()

	file = open(filename, "w", encoding="utf-8")
	json.dump(result, file, indent=2, ensure_ascii=False)
	file.close()
	print("written " + filename)

	#os.system("shutdown -s -t 60")

if __name__ == "__main__":
	main()
