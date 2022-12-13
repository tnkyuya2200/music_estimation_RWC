import myfunctions as fn
from tqdm import tqdm
from spleeter.separator import Separator
import tensorflow as tf
import sys, os
import warnings
import numpy as np
import logging
import librosa

def main():
    tf.compat.v1.disable_eager_execution()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    

    db = fn.Database(sys.argv[1])
    ID = int(sys.argv[2])
    m = db.load_Music_by_ID(ID)
    m.y , m.sr = librosa.load(m.FilePath, mono=False)
    m.y = librosa.util.normalize(m.y, axis=1)

    esti_vocals = np.empty(m.y.shape)
    esti_acc = np.empty(m.y.shape)
    separator = Separator("spleeter:4stems")
    sep = 20000000
    for i in range(0, m.y.shape[1], sep):
        prediction = separator.separate(m.y[:, i:min(i+sep, m.y.shape[1])].T)
        esti_vocals[:, i:min(i+sep, m.y.shape[1])] = prediction["vocals"].T
        esti_acc[:, i:min(i+sep, m.y.shape[1])] = prediction["other"].T

    query = """
    UPDATE features SET esti_vocals = ? , esti_acc = ?
    where ID == ?;
    """
    data = (esti_vocals, esti_acc, m.ID)
    db.cur.execute(query, data)

if __name__ == "__main__":
    main()
