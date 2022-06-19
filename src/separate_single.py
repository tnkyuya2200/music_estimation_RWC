import myfunctions as fn
from tqdm import tqdm
from spleeter.separator import Separator
import tensorflow as tf
import sys
import warnings
import numpy as np

def main():
	tf.compat.v1.disable_eager_execution()

	db = fn.Database(sys.argv[1])
	ID = int(sys.argv[2])
	m = db.load_Music_by_ID(ID)

	esti_vocals = np.empty(m.y.shape)
	esti_acc = np.empty(m.y.shape)
	separator = Separator("spleeter:4stems")
	sep = 20000000
	for i in range(0, m.y.shape[1], sep):
		prediction = separator.separate(m.y[:, i:min(i+sep, m.y.shape[1])].T)
		esti_vocals[:, i:min(i+sep, m.y.shape[1])] = prediction["vocals"].T
		esti_acc[:, i:min(i+sep, m.y.shape[1])] = prediction["other"].T

	query = """
	UPDATE music SET esti_vocals = ? , esti_acc = ?
	where ID == ?;
	"""
	data = (esti_vocals, esti_acc, m.ID)
	db.cur.execute(query, data)

if __name__ == "__main__":
	main()