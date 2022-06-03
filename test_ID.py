import myfunctions as fn
from tqdm import tqdm
import sys

db = fn.Database("music_light.db")

x = db.load_ID(sys.argv[1])
m = fn.Music()
m.load_database(x)
print(m.y.shape)
