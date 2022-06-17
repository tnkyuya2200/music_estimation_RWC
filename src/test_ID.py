import myfunctions as fn
from tqdm import tqdm
import sys

db = fn.Database("music_light.db")

x = db.load_ID(sys.argv[1])
m = fn.Music()
m.load_database(x)
print(m.y.shape)
for i in range(0, m.y.shape[1], 20000000):
	print(i, ":", min(i+20000000, m.y.shape[1]))