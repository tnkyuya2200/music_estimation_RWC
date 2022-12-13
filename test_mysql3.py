import mysql.connector
from mysql.connector import errorcode
import json
import numpy as np
cnx = None
try:
    cnx = mysql.connector.connect(
        user="root",
        host="localhost",
        database="python_db",
        unix_socket="/data/mysql/mysql.sock"
    )

    if cnx.is_connected:
        print("OK")

    cur = cnx.cursor()
    sql = """
    select data from test
    """
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        data = json.loads(row[0])
        print(type(data), data)
        data2 = np.array(data)
        print(type(data2), data2)
    cur.close()
except Exception as e:
    print("error Occured:", e)

finally:
    if cnx is not None and cnx.is_connected():
        cnx.close()
