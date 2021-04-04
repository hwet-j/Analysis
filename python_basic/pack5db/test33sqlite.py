# 개인용 데이터 베이스 SQLite3 : python에 기본 설치 되어있음

import sqlite3
print(sqlite3.sqlite_version_info)

# conn = sqlite3.connect('example.db')    
conn = sqlite3.connect(':memory:')  # ram에 DB 생성 - 테스트용으로 연습할 때

try:
    cur = conn.cursor()
    
    cur.execute('create table if not exists friends(name text, phone text)')
    
    cur.execute("insert into friends values('홍길동','111-1111')")
    cur.execute("insert into friends values(?,?)",('tom', '222-2222'))
    conn.commit()
    
    cur.execute("select * from friends")
    
    for f in cur:
        print(f[0] + " " + f[1])
        
except Exception as e:
    print(e)
    conn.rollback()
finally:
    conn.close()