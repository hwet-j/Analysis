# localdb : sqlite - db자료 -> DataFrame -> db

import sqlite3


sql = """create table if not exists test(product varchar(10), 
maker varchar(10), weight real, price integer)
"""

conn = sqlite3.connect(':memory:')
#conn = sqlite3.connect('mydb')
conn.execute(sql)

data = [('mouse', 'sam', 12.5, 6000), ('keyboard','lg',502.0,86000)]
stmt = "insert into test values(?,?,?,?)"
conn.executemany(stmt, data)
conn.commit()

data1 = ('연필','모나미',3.5,500)
cursor = conn.execute("select * from test")
conn.execute(stmt, data1)
rows = cursor.fetchall()
for a in rows:
    print(a)

print()    
# DataFrame에 저장 1 - cursor.fetchall() 이용
import pandas as pd
#df1 = pd.DataFrame(rows, columns = ['product','maker','weight','price'])
print(*cursor.description)
df1 = pd.DataFrame(rows, columns = list(zip(*cursor.description))[0])
print(df1)

# DataFrame에 저장 2 - pd.read_sql 이용
print()
df2 = pd.read_sql("select * from test", conn)  
print(df2)
print()
print(df2.to_html())

print('------------------')
# DataFrame의 자료를 DB로 저장
data = {
    'irum':['신선해','신기해','신기한'],
    'nai':[22,25,27],
}
frame = pd.DataFrame(data)
print(frame())

print()
conn = sqlite3.connect('test.db')
frame.to_sql("mytable",conn, if_exists = 'append', index=False)
df3 = pd.read_sql('select * from mytable', conn)
print(df3)

cursor.close()
conn.close()





    

