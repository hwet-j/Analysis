# 원격 DB 연동 후 jikwon 자료를 읽어 DataFrame에 저장

import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import csv
import ast
import sys

try:
    with open('mariadb.txt', 'r') as f:
        config = f.read()
        
except Exception as e:
    print("read err : ", e)
    sys.exit()
    
    
config = ast.literal_eval(config)
print(config)

print()
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql="""
    select jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay
    from jikwon inner join buser
    on jikwon.buser_num = buser.buser_no
    """
    cursor.execute(sql)
    
    for (jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay) in cursor:
        print(jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_gen, jikwon_pay)
    
    # jikwon.csv 파일로 저장
    '''
    with open('jikwon.csv', 'w', encoding='utf-8') as fw:
        writer = csv.writer(fw)
        for row in cursor:
            writer.writerow(row)
        print('저장 성공')
    '''  
        
    # csv 파일 읽기 1  
    '''
    df1 = pd.read_csv('jikwon.csv', header=None, names=('번호','이름','직급','부서','성별','연봉'))
    print(df1.head(3))
    print(df1.shape)    # (30, 6)
    '''
        
    # csv 파일 읽기 2
    df2 = pd.read_sql(sql, conn)
    df2.columns = ('번호','이름','직급','부서','성별','연봉')
    print(df2.head(3))
    
    print()
    print('건수 : ',len(df2))
    print('건수 : ', df2['이름'].count())
    
    print()
    print('직급별 인원수  : \n', df2['직급'].value_counts())
    
    print()
    print('연봉 평균  : \n', df2.loc[:,'연봉'].sum()/len(df2))
    print('연봉 평균  : \n', df2.loc[:,'연봉'].mean())
    
    print()
    print('연봉 요약 통계 : \n', df2.loc[:,'연봉'].describe())
    
    print()
    print('연봉이 8000 이상 : \n', df2.loc[df2['연봉'] >= 8000])
    
    print()
    print('연봉이 5000 이상인 영업부 : \n', df2.loc[(df2['연봉'] >= 5000) & (df2['부서'] == '영업부')])
    
    # 직급별 성 비율
    print('crosstab---------')
    ctab = pd.crosstab(df2['성별'], df2['직급'], margins=True)
    print(ctab)
    
    print('groupby --------')
    print(df2.groupby(['성별', '직급'])['이름'].count())
    
    print('pivot_table-----------')
    print(df2.pivot_table(['연봉'], index=['성별'], columns=['직급'], aggfunc = np.mean))
    
    # 시각화
    # 직급별 연봉 평균
    jik_ypay = df2.groupby(['직급'])['연봉'].mean()
    print(jik_ypay, type(jik_ypay))
    print(jik_ypay.index)
    print(jik_ypay.values)
    
    plt.pie(jik_ypay, labels = jik_ypay.index,
            labeldistance = 0.5,
            counterclock=False,
            shadow=True,
            explode=(0.2, 0, 0, 0.3, 0))
    
    plt.show()
    
    
except Exception as e:
    print('process err : ', e)
finally:
    cursor.close()
    conn.close()


print('------------------')
# DataFrame의 자료를 DB로 저장
data = {
    'irum':['tom','james','john'],
    'nai':[22,25,27],
}
frame = pd.DataFrame(data)
print(frame)

print()
# pip install pymysql
# pip install sqlalchemy

from sqlalchemy import create_engine

# MySQL Connector using pymysql
import pymysql
pymysql.install_as_MySQLdb()

engine = create_engine("mysql+mysqldb://root:"+"123"+"@localhost/test", encoding="utf-8")
conn = engine.connect()

# MySQL에 저장하기
# pandas의 to_sql() 함수 사용저장
frame.to_sql(name = "mytable",con = engine, if_exists = 'append', index=False)
df3 = pd.read_sql('select * from mytable', conn)
print(df3)






