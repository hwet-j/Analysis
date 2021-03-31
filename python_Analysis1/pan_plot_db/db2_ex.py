#pandas 문제 5)
#  MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#      - DataFrame의 자료를 파일로 저장
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#      - 부서명, 직급으로 교차테이블을 작성(crosstab)
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력

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
# print(config)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    cursor2 = conn.cursor()
    sql="""
    select jikwon_no, jikwon_name, jikwon_jik, buser_name, jikwon_pay
    from jikwon inner join buser
    on jikwon.buser_num = buser.buser_no
    """
    sql2 = """
    select gogek_no, gogek_name, gogek_tel, gogek_damsano, jikwon_name
    from gogek inner join jikwon
    on jikwon.jikwon_no = gogek.gogek_damsano
    """
    cursor.execute(sql)
    cursor2.execute(sql2)
    
    # DataFrame
    data = pd.DataFrame(cursor, columns=['사번','이름','직급','부서','연봉'])
    data2 = pd.DataFrame(cursor2, columns=['번호','이름','전화','담당','직원'])
    # print(data)
    # print(data2)
    
    # 부서별 연봉 합,최대,최소
    print('부서별 연봉합  : ', data.groupby(['부서'])['연봉'].sum())
    print()
    print('부서별 연봉최대  : ', data.groupby(['부서'])['연봉'].max())
    print()
    print('부서별 연봉최소  : ', data.groupby(['부서'])['연봉'].min())
    
    # 부서, 직급으로 교차테이블작성 (crosstab)
    print('\n부서, 직급으로 교차테이블작성 (crosstab)')
    ctab = pd.crosstab(data['부서'], data['직급'], margins=True)
    print(ctab)
    
    # 직원별 담당 고객자료(고객번호, 고객명, 고객전화)
    print('\n직원별 담당 고객자료(고객번호, 고객명, 고객전화)')
    print(pd.pivot_table(data2, index=['직원','번호','이름','전화']))
    
    
    
    # 부서별 연봉의 평균 가로막대 그래프 - y축 값?
    bu_ypay = data.groupby(['부서'])['연봉'].mean()
    plt.barh(['관리부','영업부','전산부','총무부'],bu_ypay)
    plt.show()
    
except Exception as e:
    print('process err : ', e)
finally:
    cursor.close()
    conn.close()




