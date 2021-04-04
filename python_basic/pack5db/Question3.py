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
    
    jikwon_jik = input('직급입력 :')
    
    sql = """
    select jikwon_no, jikwon_name, jikwon_jik, buser_num, jikwon_pay
    from jikwon
    where jikwon_jik='{0}'
    """.format(jikwon_jik)
    
    cursor.execute(sql)
    datas = cursor.fetchall()
    
    if len(datas) == 0:
        print(str(jikwon_jik) + '라는 직급을 가진 직원은 없어요')
        sys.exit()
        
    print("사번\t직원명\t직급\t부서번호\t연봉")
    for d in datas:
        print(d[0],"\t", d[1],"\t", d[2],"\t", d[3],"\t", d[4])
    
    print('인원수:' + str(len(datas)))
    
except Exception as e:
    print('err : ' + str(e))
finally:
    cursor.close()
    conn.close()
