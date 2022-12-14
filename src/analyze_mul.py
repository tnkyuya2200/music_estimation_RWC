import myfunctions as fn
from tqdm import tqdm
import sys
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

db = fn.Database()
def task(ID):
    m = fn.Database().load_Music_by_ID(ID)
    m.load_and_analyze_music()
    print(m.melody)
    return (
        m.y, m.beats, m.bpm, m.frame_size,
        m.quantize, m.melody, m.chords, m.ID
    )

def main():
    db = fn.Database()
    IDs = db.getIDlist()
    with ProcessPoolExecutor(max_workers=8) as executor:
        with tqdm(total=len(IDs[1:])) as progress:
            futures = []
            for ID in IDs[1:]:
                future = executor.submit(
                    task, ID
                    )
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
        data = [f.result() for f in futures]
    query = """
UPDATE features SET 
    y = %s,
    sr = %s,
    bpm = %s,
    frame_size = %s,
    quantize = %a
WHERE ID = %s
    """
    db.cur.executemany(
        query,
        [(m.y, m.sr, m.bpm, m.frame_size, m.quantize, m.ID) for m in data]
    )
    for music in data:
        db.update_data("beats", music.ID, music.beats)
        db.update_data("melody", music.ID, music.melody)
        db.update_data("chords", music.ID, music.chords)
    db.con.commit()
if __name__ == "__main__":
    main()