import mysql.connector
from mysql.connector import errorcode

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
    CREATE TABLE test(
        id INT NOT NULL PRIMARY KEY,
        data JSON
    )"""
    cur.execute(sql)

    cur.execute("SHOW TABLES")
    print(cur.fetchall())

    cur.close()
except Exception as e:
    print("error Occured:", e)

finally:
    if cnx is not None and cnx.is_connected():
        cnx.close()
