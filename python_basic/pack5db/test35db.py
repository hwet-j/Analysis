# 키보드로 부서번호를 입력받아 해당 부서에 근무하는 직원 출력

import MySQLdb
import sys

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
    cursor = conn.cursor()
    
    buser_no = input('부서번호 입력 :')
    # 길어질수도 있는 sql문을 주석 처럼 작성해줄수있다.
    sql = """
    select jikwon_no, jikwon_name, jikwon_jik, jikwon_pay
    from jikwon
    where buser_num={0}
    """.format(buser_no)
    
    cursor.execute(sql)
    datas = cursor.fetchall()
    
    if len(datas) == 0:
        print(str(buser_no) + '번 부서는 없어요')
        sys.exit()
        
    for d in datas:
        print(d[0], d[1], d[2], d[3])
    
    print('인원수:' + str(len(datas)))
    
except Exception as e:
    print('err : ' + str(e))
finally:
    cursor.close()
    conn.close()