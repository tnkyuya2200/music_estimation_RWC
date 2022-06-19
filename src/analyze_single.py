import myfunctions as fn
from tqdm import tqdm
import sys
import warnings

def main():
	warnings.simplefilter('error')
	db = fn.Database(sys.argv[1])
	ID = int(sys.argv[2])
	m = db.load_Music_by_ID(ID)
	m.analyze_music()
	query = """
	UPDATE music SET beats = ?, bpm = ?, frame_size = ?, quantize = ?, melody = ?, chords = ?
	where ID == ?;
	"""
	data = (m.beats, m.bpm, m.frame_size, m.quantize, m.melody, m.chords, m.ID)
	db.cur.execute(query, data)
	print("ID:" + str(ID) + " analyzed completed!")

if __name__ == "__main__":
	main()