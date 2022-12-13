import mysql.connector
from mysql.connector import errorcode
import numpy as np
import json
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

    data = np.array([1,2,3,4,5])
    data2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(data)
    print(data2)

    sql = """
    INSERT INTO test
        (ID, data)
    VALUES
        (%s, %s)
    """
    cur.execute(sql, (2, json.dumps(data2.tolist())))
    cnx.commit()
    cur.close()
except Exception as e:
    print("error Occured:", e)

finally:
    if cnx is not None and cnx.is_connected():
        cnx.close()
