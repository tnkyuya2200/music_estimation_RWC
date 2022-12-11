import myfunctions as fn
import sys

def main():
	db = fn.Database(sys.argv[1])
	m = fn.Music()
	m.load_music(sys.argv[2])
	query = """
UPDATE features SET FilePath = %s where ID == %s
"""
	data = (m.FilePath, m.ID)
	db.cur.execute(query, data)
	db.con.commit()
if __name__ == "__main__":
	main()
