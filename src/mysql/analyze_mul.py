import myfunctions as fn
from tqdm import tqdm
import sys
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

db = fn.Database("music.db")
def task(m):
	m.load_and_analyze_music()
	return (m.y, m.sr, m.beats, m.bpm, m.frame_size, m.quantize, m.melody, m.chords, m.ID)

def main():
	db = fn.Database(sys.argv[1])
	IDs = db.getIDlist()
	query = """
	UPDATE features SET y = %s, sr = %s, beats = %s, bpm = %s, frame_size = %s, quantize = %s, melody = %s, chords = %s
		where ID == %s
	"""
	with ProcessPoolExecutor(max_workers=8) as executor:
		with tqdm(total=len(IDs[1:])) as progress:
			futures = []
			for ID in IDs[1:]:
				future = executor.submit(
					task, db.load_Music_by_ID(ID)
					)
				future.add_done_callback(lambda p: progress.update())
				futures.append(future)
		data = [f.result() for f in futures]
	db.cur.executemany(query, [f.result() for f in futures])
	db.con.commit()

if __name__ == "__main__":
    main()
