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
from typing import Optional
warnings.simplefilter("error")

def tqdm_joblib(total: Optional[int] = None, **kwargs):

	pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __call__(self, *args, **kwargs):
			pbar.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

	try:
		yield pbar
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		pbar.close()

def task(ID):
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
	return tmp_dict

def main():
	db = fn.Database(sys.argv[1])
	IDs = db.getIDlist()

	result = {"test_file": sys.argv[2]}
	result["db"] = [] 
	filename = os.path.join(sys.argv[3], os.path.splitext(os.path.basename(sys.argv[2]))[0] + ".json")
	test_music = db.load_Music_by_ID(0)
	test_music.analyze_music(4)
	test_music_q2 = db.load_Music_by_ID(0)
	test_music_q2.analyze_music(2)
	with tqdm_joblib():
		results = joblib.Parallel(n_jobs=-1)(delayed(task)(i) for ID in IDs[1:])
	for result in results:
		result["db"].append(tmp_dict)

	result["timestamp"] = datetime.now().isoformat()

	file = open(filename, "w", encoding="utf-8")
	json.dump(result, file, indent=2, ensure_ascii=False)
	file.close()
	print("written " + filename)

	#os.system("shutdown -s -t 60")

if __name__ == "__main__":
	main()