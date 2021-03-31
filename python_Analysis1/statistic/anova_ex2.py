# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오.
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

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
    select buser_name, jikwon_pay
    from jikwon inner join buser
    on jikwon.buser_num = buser.buser_no
    """
    
    cursor.execute(sql)
    
     # 데이터 읽기     
    df = pd.read_sql(sql, conn)
    df.columns = ('부서명','연봉')
    
    Affairs = df['부서명'] == '총무부'
    Sales = df['부서명'] == '영업부'
    IT = df['부서명'] == '전산부'
    Management = df['부서명'] == '관리부'
    
    affairs = df[Affairs]
    sales = df[Sales]
    it = df[IT]
    management = df[Management]
    
    print
    # 1. 정규성 확인
    print(stats.shapiro(affairs.연봉).pvalue)   # pvalue=0.868040
    print(stats.shapiro(sales.연봉).pvalue)   # pvalue=0.592393
    print(stats.shapiro(it.연봉).pvalue)   # pvalue=0.486010
    print(stats.shapiro(management.연봉).pvalue)   # pvalue=0.416217
    print('정규성확인 : ', stats.ks_2samp(affairs, sales))   # pvalue=0.930735 > 0.05 정규성 만족
    print('정규성확인 : ', stats.ks_2samp(affairs, it))   # pvalue=0.923809
    print('정규성확인 : ', stats.ks_2samp(affairs, management))   # pvalue=0.552380
    print('정규성확인 : ', stats.ks_2samp(sales, it))   # pvalue=0.923809
    print('정규성확인 : ', stats.ks_2samp(sales, management))   # pvalue=0.552380
    print('정규성확인 : ', stats.ks_2samp(it, management))   # pvalue=0.771428
    
    # 2. 등분산성 확인
    #print(stats.levene(affairs.연봉, sales.연봉).pvalue) # 0.915044305043978> 0.05 등분산성 만족

    #print(np.mean(affairs))     # 5414.285714
    #print(np.mean(sales))       # 4908.333333
    
    #data2 = stats.ttest_ind(affairs.연봉, sales.연봉, equal_var=True) 
    #print('t-통계량: ', data2[0])  # t-통계량:  0.4585177708256519
    #print('p-value: ', data2.pvalue) # p-value:  0.6523879191675446
    # 정규성 성립안해서 wilcoxon으로 분석해야하지만 여기서는 t-test검정으로 함
    # data2 = stats.wilcoxon(df1.jikwon_pay, df2.jikwon_pay)
    # print(data2) # process err:  The samples x and y must have the same length.
    # 결론: p-value 0.6523879191675446 > 0.05이므로 귀무가설 채택, 대립가설 기각
    # 총무부, 영업부 직원의 연봉의 평균에 차이없다
    
    #print()
    #print(stats.ttest_ind(affairs['연봉'], sales['연봉'], equal_var = True))
    # pvalue = 0.652387 > 0.05이므로 총무부, 영업부 직원의 연봉의 평균에 차이가 없다.
    
    # Mann-Whitney U 검정
    # 두 개의 독립된 집단 간의 특정 값의 평균 비교
    # 이떄, 두 개의 독립된 집단이 정규분포를 따르지 않을 때 사용한다.
    #data_mann = stats.mannwhitneyu(affairs['연봉'], sales['연봉'])
    #print('mannwhitneyu : ', data_mann) # pvalue=0.2360667
    
    
except Exception as e:
    print('process err : ', e)
finally:
    cursor.close() 
    conn.close()


