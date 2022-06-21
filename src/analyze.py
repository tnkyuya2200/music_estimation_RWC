import myfunctions as fn
from tqdm import tqdm
import sys

if len(sys.argv) == 1:
    db = fn.Database("music.db")
else:
    db = fn.Database(sys.argv[1])

for ID in tqdm(range(1,db.getdbsize()+1, leave=False, position=1)):
    m = db.load_Music_by_ID(ID)
    m.analyze()
    query = """
UPDATE music SET y = ?, sr = ?, beats = ?, bpm = ?, melody = ?, acc = ?, chords = ?
    where ID == ?;
    """
    data = (m.y, m.sr, m.beats, m.bpm, m.melody, m.acc, m.chords, m.ID)
    db.cur.execute(query, data)
