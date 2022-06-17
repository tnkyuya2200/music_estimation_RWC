import myfunctions as fn
from tqdm import tqdm
import sys
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

db = fn.Database("music.db")
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] &(message)s"))
    logger = getLogger()
    logger.addHandler(handler)
    logger.setLevel(INFO)

def task(ID):
    x = db.load_ID(ID)
    m = fn.Music()
    m.load_database(x)
    m.analyze()
    return (m.beats, m.bpm, m.melody, m.acc, m.chords, m.ID)

def main():
    if len(sys.argv) == 1:
        db = fn.Database("music.db")
    else:
        db = fn.Database(sys.argv[1])
    getLogger().info("Database loaded")
    query = """
UPDATE music SET beats = ?, bpm = ?, melody = ?, acc = ?, chords = ?
    where ID == ?;
    """
    with ProcessPoolExecutor() as executor:
        for single_result in tqdm(executor.map(task, range(1,db.getdbsize()[0]+1))):
            print(single_result)
            db.cur.execute(query, single_result)


if __name__ == "__main__":
    main()