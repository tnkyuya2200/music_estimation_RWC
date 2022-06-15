import myfunctions as fn
from tqdm import tqdm
import sys

if len(sys.argv) == 1:
    db = fn.Database("music.db")
else:
    db = fn.Database(sys.argv[1])

for ID in tqdm(range(1,db.getdbsize()[0]+1)):
    x = db.load_ID(ID)
    m = fn.Music()
    m.load_database(x)
    m.analyze()
    query = """
UPDATE music SET beats = ?, bpm = ?, melody = ?, acc = ?, chords = ?
    where ID == ?;
    """
    data = (m.beats, m.bpm, m.melody, m.acc, m.chords, m.ID)
    db.cur.execute(query, data)
