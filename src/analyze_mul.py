import myfunctions as fn
from tqdm import tqdm
import sys
from logging import StreamHandler, Formatter, INFO, getLogger
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

db = fn.Database("music.db")
def task(m):
	m.analyze_music()
	return (m.y, m.sr, m.beats, m.bpm, m.frame_size, m.quantize, m.melody, m.chords, m.ID)

def main():
	db = fn.Database(sys.argv[1])
	IDs = db.getIDlist()
	query = """
	UPDATE music SET y = ?, sr = ?, beats = ?, bpm = ?, frame_size = ?, quantize = ?, melody = ?, chords = ?
		where ID == ?;
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

if __name__ == "__main__":
    main()
