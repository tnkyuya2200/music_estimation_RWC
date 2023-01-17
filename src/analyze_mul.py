import myfunctions as fn
from tqdm import tqdm
import sys
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

db = fn.Database("music.db")
query = """
    UPDATE features SET y = ?, sr = ?, beats = ?, bpm = ?, frame_size = ?, quantize = ?, melody = ?, chords = ?, fingerprint = ?
        where ID == ?;
"""
def task(ID):
    print("ID:", ID, "started")
    db = fn.Database(sys.argv[1])
    m = db.load_Music_by_ID(ID)
    m.load_and_analyze_music()
    data = (m.y, m.sr, m.beats, m.bpm, m.frame_size, m.quantize, m.melody, m.chords, m.fingerprint, m.ID)
    db.cur.execute(query, data)
    print("\tID:", ID, "ended")


def main():
    db = fn.Database(sys.argv[1])
    IDs = db.getIDlist()
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for ID in IDs[1:]:
            future = executor.submit(
                task, ID
                )
            futures.append(future)
        #data = [f.result() for f in futures]
    #db.cur.executemany(query, data)

if __name__ == "__main__":
    main()
