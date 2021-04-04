# 원격 DB 연동 : Maria DB
'''
import MySQLdb
conn = MySQLdb.connect(host = '127.0.0.1', user = 'root', password='123', database='test')
print(conn)
conn.close
'''

# sangdata table 자료 읽기
import MySQLdb

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor() # SQL문 처리용
    
    # 자료 추가 
    #sql = "insert into sangdata(code, sang, su, dan) values(17, '새우깡','5','1200')"
    #cursor.execute(sql)
    #conn.commit()
    '''
    sql = "insert into sangdata values(%s,%s,%s,%s)"
    sql_data = '10','감자깡',10,2000
    cou = cursor.execute(sql, sql_data) # insert 성공하면 1 실피하면 0을 반환
    #print(cursor)
    conn.commit()
    '''
    
    # 자료 수정 
    '''
    sql = "update sangdata set sang=%s,su=%s,dan=%s where code=%s"
    sql_data = ('파래깡',7,2200,10)
    cou = cursor.execute(sql, sql_data)     # update 성공하면 1 실피하면 0을 반환
    print('update 성공 건수:',cou)
    conn.commit()
    '''
    '''
    # 자료 삭제
    code = '10'
    # 비권장
    # sql = "delete from sangdata where code="+code    
    # cursor.execute(sql)
    
    # 권장 1
    # sql = "delete from sangdata where code=%s"
    # cursor.execute(sql, (code,))    # 뒤에 콤마가 있어야 튜플 타입임. 
    
    # 권장 2
    sql = "delete from sangdata where code='{0}'".format(code)   # 숫자는 '' 가 없어도되지만 문자는 존재해야함
    cursor.execute(sql)
    conn.commit()
    '''
    
    
    
    # 자료 읽기
    sql = "select code,sang,su,dan from sangdata"
    cursor.execute(sql) # DB의 자료를 읽어 cursor 객체가 그 결과를 기억
    
    print('읽기1')
    for data in cursor.fetchall():
        #print(data)
        print('%s %s %s %s'%data)
        
    print('읽기2')
    for r in cursor:
        print(r[0], r[1], r[2], r[3])
        
    print('읽기3')
    for (code,sang,su,dan) in cursor:
        print(code,sang,su,dan)
        
    print('읽기3-1')  # 컬럼에있는 변수명을 가져오는게 아닌 내가 지정해준 변수일뿐(가독성을위해 똑같이 해줄뿐)
    for (ab,cd,수량,단가) in cursor:
        print(ab,cd,수량,단가)        
        
except MySQLdb.connections.Error as err:
    print('db err : ' + str(err))
    conn.rollback()
except Exception as e:
    print('err : ' + str(e))
finally:
    cursor.close()
    conn.close()