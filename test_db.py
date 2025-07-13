import pymysql

db = pymysql.connect(
    host="localhost",
    user="root",
    password="1234",  # replace with your password
    database="carebot_users",
    cursorclass=pymysql.cursors.DictCursor
)

cursor = db.cursor()
cursor.execute("SHOW TABLES;")

for table in cursor.fetchall():
    print(table)

db.close()

