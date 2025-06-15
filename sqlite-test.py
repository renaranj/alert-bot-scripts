import sqlite3
conn = sqlite3.connect('test.db')
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS users(
   userid INT PRIMARY KEY,
   fname TEXT,
   lname TEXT,
   gender TEXT);
""")
conn.commit()
cur.execute("""INSERT INTO users(userid, fname, lname, gender) 
   VALUES('00001', 'Nik', 'Piepenbreier', 'male');""")
conn.commit()
cur.execute("SELECT * FROM users;")
all_results = cur.fetchall()
print(all_results)
