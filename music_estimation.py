#coding:utf-8
import sqlite3
import json
import myfunctions as fn
import sys, os
import json
from datetime import datetime
from tqdm import tqdm

result = {}
filename = os.path.join("result", "result_tmp2", os.path.split(os.path.dirname(sys.argv[1]))[1], "result.json")

for arg in tqdm(sys.argv[1:], leave=False, position=1):
    result[arg] = fn.compare_all(test_path=arg)
result["timestamp"] = datetime.now().isoformat()

file = open(filename, "w", encoding="utf-8")
json.dump(result, file, indent=2, ensure_ascii=False)
file.close()
print("written " + filename)

#os.system("shutdown -s -t 60")