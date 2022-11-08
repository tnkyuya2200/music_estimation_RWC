import myfunctions as fn
from tqdm import tqdm
import sys
import warnings

def main():
	warnings.simplefilter('error')
	db = fn.Database(sys.argv[1])
	ID = int(sys.argv[2])
	m = db.load_Music_by_ID(ID)
	m.load_and_analyze_music()
	query = """
	UPDATE music SET y = ?, sr = ?, beats = ?, bpm = ?, frame_size = ?, quantize = ?, melody = ?, chords = ?
	where ID == ?;
	"""
	data = (m.y, m.sr, m.beats, m.bpm, m.frame_size, m.quantize, m.melody, m.chords, m.ID)
	db.cur.execute(query, data)

if __name__ == "__main__":
	main()
