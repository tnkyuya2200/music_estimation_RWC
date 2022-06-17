import myfunctions as fn
from tqdm import tqdm
import sys

def main():
	db = fn.Database(sys.argv[1])
	ID = int(sys.argv[2])
	m = db.load_Music_by_ID(ID)
	m.separate_music()
	query = """
	UPDATE music SET esti_vocals = ? , esti_acc = ?
	where ID == ?;
	"""
	data = (m.esti_vocals, m.esti_acc, m.ID)
	db.cur.execute(query, data)
	print("execute completed!")

if __name__ == "__main__":
	main()