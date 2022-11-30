import myfunctions as fn
import sys

def main():
	db = fn.Database(sys.argv[1])
	m = fn.Music()
	m.load_music(sys.argv[2])
	query = """
UPDATE features SET FilePath = ? where ID == ?;
"""
	data = (m.FilePath, m.ID)
	db.cur.execute(query, data)
	
if __name__ == "__main__":
	main()
