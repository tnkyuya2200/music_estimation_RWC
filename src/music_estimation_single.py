#coding:utf-8
import sqlite3
import json
import myfunctions as fn
import sys, os
import json
from datetime import datetime
from tqdm import tqdm

result = {}
filename = os.path.join("tmp.json")
db = fn.Database(sys.argv[1])
result = fn.compare_all(db.load_Music_by_ID(0), db)
result["timestamp"] = datetime.now().isoformat()

file = open(filename, "w", encoding="utf-8")
json.dump(result, file, indent=2, ensure_ascii=False)
file.close()
print("written " + filename)

#os.system("shutdown -s -t 60")