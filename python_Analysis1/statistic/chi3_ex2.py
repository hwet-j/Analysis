#문제2) jikwon_jik과 jikwon_pay 간의 관련성 분석. 가설검정하시오.

import MySQLdb
import pandas as pd
import ast
import sys
import scipy.stats as stats

try:
    with open('mariadb.txt', 'r') as f:
        config = f.read()
        
except Exception as e:
    print("read err : ", e)
    sys.exit()
    
config = ast.literal_eval(config)
#print(config)

print()
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    
    sql="""
    select jikwon_no, jikwon_name, jikwon_jik, jikwon_gen, jikwon_pay
    from jikwon
    """
    
    cursor.execute(sql)
    '''
    for (jikwon_no, jikwon_name, jikwon_jik, jikwon_gen, jikwon_pay) in cursor:
        print(jikwon_no, jikwon_name, jikwon_jik, jikwon_gen, jikwon_pay)
    '''
    
    # 데이터 읽기     
    df = pd.read_sql(sql, conn)
    df.columns = ('사번','이름','직급','성별','연봉')
    
    # 연봉 범주화 
    df.loc[(df['연봉'] >= 1000) & (df['연봉'] < 3000),'연봉'] = 1
    df.loc[(df['연봉'] >= 3000) & (df['연봉'] < 5000),'연봉'] = 2
    df.loc[(df['연봉'] >= 5000) & (df['연봉'] < 7000),'연봉'] = 3
    df.loc[(df['연봉'] >= 7000),'연봉'] = 4
    
    '''
    # 범주화 
    jik = []
    for row in df['직급']:
        if row == '이사':
            jik.append(1)
        elif row == '부장':
            jik.append(2)
        elif row == '과장':
            jik.append(3)
        elif row == '대리':
            jik.append(4)
        elif row == '사원':
            jik.append(5)
    del df['직급']
    df['직급'] = jik
    '''
    
    ctab = pd.crosstab(index = df['직급'], columns= df['연봉'])
    ctab.columns = ['[1,3)','[3~5)','[5~7)','[7']
    print(ctab)
    
    chi, p, _, _= stats.chi2_contingency(ctab)
    print('chi :',chi)      # chi : 37.403493
    print('p-value :',p)    # p-value : 0.0202980
    
    # 해석 : p-value : 0.00019211 < 0.05 이므로 귀무가설을 기각
    # 대립 : 직급과 연봉 간의 관련이 있다. (독립이 아니다.)
    
except Exception as e:
    print('process err : ', e)
finally:
    cursor.close()
    conn.close()




