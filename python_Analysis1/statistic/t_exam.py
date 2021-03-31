import numpy as np
import scipy as sp
import scipy.stats as stats
import MySQLdb
import pandas as pd
import ast
import sys
import random

# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.

blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

# 정규성 확인
print(stats.shapiro(blue).pvalue)   # 0.510231 > 0.05 정규성 만족
print(stats.shapiro(red).pvalue)    # 0.510231 > 0.05 정규성 만족

# 등분산성 확인
print(stats.levene(blue, red).pvalue)   # 0.439164 > 0.05 등분산성 만족

data = stats.ttest_ind(blue, red, equal_var=True) 
#print(data)
print('p-value: ', data.pvalue) # p-value:  0.008316545714784403
# pvalue = 0.00831654 < 0.05이므로 귀무가설 기각. 포장지 색상에 따른 제품의 매출액에 차이가 있다.



# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.

male = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

# 비복원 추출 - 방식 2가지
random.seed(123)
# y1 = random.sample(male, 15)
# y2 = random.sample(female, 15)

np.random.seed(123)
# np.random.choice(뽑을 표본, 표본개수, replace=False) 비복원 추출
x1 = np.random.choice(male, 15, replace=False)
x2 = np.random.choice(female, 15, replace=False)

# 1. 정규성 확인
print(stats.shapiro(male).pvalue)  # 0.8990891575813293 > 0.05 정규성 만족
print(stats.shapiro(female).pvalue) # 0.020763792097568512 < 0.05 정규성 불만족

# 2. 등분산성 확인
print(stats.fligner(male, female).pvalue) # 0.08046255219706801 > 0.05 등분산성 만족

data1 = stats.ttest_ind(male, female, equal_var=True) 
print('t-통계량: ', data1[0])   # t-통계량:  -1.7661241586261762
print('p-value: ', data1.pvalue) # p-value:  0.08827773360187119
# 결론: p-value 0.08827773360187119 > 0.05 이므로 귀무가설 채택, 대립가설 기각
# 남여의 혈관 내의 콜레스테롤 양에 차이가 없다


# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

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
    #print(df)
    Affairs = df['부서명'] == '총무부'
    Sales = df['부서명'] == '영업부'
    affairs = df[Affairs]
    sales = df[Sales]
    # 1. 정규성 확인
    print(stats.shapiro(affairs.연봉).pvalue) # 0.02604489028453827 < 0.05 정규성 불만족
    print(stats.shapiro(sales.연봉).pvalue) # 0.02560843899846077 < 0.05 정규성 불만족
    
    # 2. 등분산성 확인
    print(stats.levene(affairs.연봉, sales.연봉).pvalue) # 0.915044305043978> 0.05 등분산성 만족

    #print(np.mean(affairs))     # 5414.285714
    #print(np.mean(sales))       # 4908.333333
    
    data2 = stats.ttest_ind(affairs.연봉, sales.연봉, equal_var=True) 
    print('t-통계량: ', data2[0])  # t-통계량:  0.4585177708256519
    print('p-value: ', data2.pvalue) # p-value:  0.6523879191675446
    # 정규성 성립안해서 wilcoxon으로 분석해야하지만 여기서는 t-test검정으로 함
    # data2 = stats.wilcoxon(df1.jikwon_pay, df2.jikwon_pay)
    # print(data2) # process err:  The samples x and y must have the same length.
    # 결론: p-value 0.6523879191675446 > 0.05이므로 귀무가설 채택, 대립가설 기각
    # 총무부, 영업부 직원의 연봉의 평균에 차이없다
    
    print()
    print(stats.ttest_ind(affairs['연봉'], sales['연봉'], equal_var = True))
    # pvalue = 0.652387 > 0.05이므로 총무부, 영업부 직원의 연봉의 평균에 차이가 없다.
    
    # Mann-Whitney U 검정
    # 두 개의 독립된 집단 간의 특정 값의 평균 비교
    # 이떄, 두 개의 독립된 집단이 정규분포를 따르지 않을 때 사용한다.
    data_mann = stats.mannwhitneyu(affairs['연봉'], sales['연봉'])
    print('mannwhitneyu : ', data_mann) # pvalue=0.2360667
    
    
except Exception as e:
    print('process err : ', e)
finally:
    cursor.close() 
    conn.close()


# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다. 
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?

midterm = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
final = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]

# 귀무가설: 이 학급의 학업능력이 변화 안했다
# 대립가설: 이 학급의 학업능력이 변화했다

print('중간고사 평균: ', np.mean(midterm)) # 74.16666666666667
print('기말고사 평균: ', np.mean(final)) # 81.66666666666667
print('두 시험의 평균의 차이: ', np.mean(midterm)-np.mean(final)) # -7.5

# 1. 정규성 확인
print(stats.shapiro(midterm).pvalue) # 0.3681465983390808 > 0.05 정규성 만족
print(stats.shapiro(final).pvalue) # 0.19300280511379242 > 0.05 정규성 만족

data3 = stats.ttest_rel(midterm, final)
print('t-통계량: ', data3[0])  # t-통계량:  -2.6281127723493993
print('p-value: ', data3.pvalue) # p-value:  0.023486192540203194
# 결론: p-value 0.023486192540203194 < 0.05이므로 귀무가설 기각, 대립가설 채택
# 이 학급의 학업능력이 변화했다













