# jikwon 테이블 자료로 chi2, t검정, ANOVA
import MySQLdb
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


import matplotlib.pyplot as plt

try:
    with open('mariadb.txt', 'r') as f:
        config = f.read()
        
except Exception as e:
    print("read err : ", e)

config = ast.literal_eval(config)

conn = MySQLdb.connect(**config)
cursor = conn.cursor()

print('교차분석(이원 카이제곱 검정 : 각 부서-범주화와 직원평가점수-범주형 간의 관련성 분석)-------')
# 독립변수 : 범주형 , 종속변수 : 범주형
# 귀무 : 각 부서와 직원평가점수 간에 관련이 없다.
# 대립 : 각 부서와 직원평가점수 간에 관련이 있다.


df = pd.read_sql("select * from jikwon", conn)
print(df.head(3))
buser = df['buser_num']
rating = df['jikwon_rating']

ctab = pd.crosstab(buser, rating)   # 교차표 작성
print(ctab)

chi, p, df, exp = stats.chi2_contingency(ctab)
print('chi:{}, p:{}, df:{}'.format(chi, p, df))
# chi:7.339285714285714, p:0.2906064076671985, df:6
# p:0.29060 > 0.05 이므로 귀무가설 : 각 부서와 직원평가점수 간에 관련이 없다.

print('교차분석(이원 카이제곱 검정 : 각 부서-범주화와 직급-범주형 간의 관련성 분석)-------')
# 귀무 : 각 부서와 직급 간에 관련이 없다.
# 대립 : 각 부서와 직급 간에 관련이 있다.

df2 = pd.read_sql("select buser_num, jikwon_jik from jikwon", conn)
print(df2.head(3))
buser = df2.buser_num
jik = df2.jikwon_jik

ctab2 = pd.crosstab(buser, jik)   # 교차표 작성
print(ctab2)

chi, p, df, exp = stats.chi2_contingency(ctab2)
print('chi:{}, p:{}, df:{}'.format(chi, p, df))
# chi:9.620617477760335, p:0.6492046290079438, df:12
# p:0.64920462 > 0.05 이므로 귀무가설 : 각 부서와 직급 간에 관련이 없다.

print('\n차이분석(t-test : 10,20번 부서-범주형과 평균연봉 값의 차이를 검정)---------')
# 독립변수 : 범주형 , 종속변수 : 연속형
# 귀무 : 두 부서 간 연봉 평균에 차이가 없다.
# 대립 : 두 부서 간 연봉 평균에 차이가 있다.
df = pd.read_sql("select buser_num, jikwon_pay from jikwon where buser_num in(10,20)", conn)
df_10 = pd.read_sql("select buser_num, jikwon_pay from jikwon where buser_num=10", conn)
df_20 = pd.read_sql("select buser_num, jikwon_pay from jikwon where buser_num=20", conn)
buser10 = df_10['jikwon_pay']
buser20 = df_20['jikwon_pay']

print('평균 : ', np.mean(buser10), ' ', np.mean(buser20)) # 5414.285714285715   4908.333333333333
t_result = stats.ttest_ind(buser10, buser20)
print(t_result)
# Ttest_indResult(statistic=0.4585177708256519, pvalue=0.6523879191675446)
# pvalue=0.652387 > 0.05 귀무가설 채택 : 두 부서 간 연봉 평균에 차이가 없다.

print('\n분산분석(ANOVA : 각 부서( 부서라는 1개의 요인에 4개 그룹이 존재) - 범주형과 평균연봉 값의 차이를 검정)--------')
# 독립변수 : 범주형 , 종속변수 : 연속형
# 귀무 : 4개 부서 간 연봉 평균에 차이가 없다.
# 대립 : 4개 부서 간 연봉 평균에 차이가 있다.
df3 = pd.read_sql("select buser_num, jikwon_pay from jikwon", conn)
buser = df3['buser_num']
pay = df3['jikwon_pay']

gr1 = df3[df3['buser_num'] == 10]['jikwon_pay']
gr2 = df3[df3['buser_num'] == 20]['jikwon_pay']
gr3 = df3[df3['buser_num'] == 30]['jikwon_pay']
gr4 = df3[df3['buser_num'] == 40]['jikwon_pay']
print(gr1)

# 시각화 
plt.boxplot([gr1, gr2, gr3, gr4])
#plt.show()

# 방법1
f_sta, pv = stats.f_oneway(gr1, gr2, gr3, gr4)
print('f값 : ', f_sta)   # f값 :  0.41244077
print('p값 : ', pv)      # p값 :  0.74544218 > 0.05 귀무채택 : 4개 부서 간 연봉 평균에 차이가 없다.

# 방법 2
lmodel = ols('jikwon_pay ~ C(buser_num)', data = df3).fit()
result = anova_lm(lmodel, type = 2)
print(result)   # 0.745442 > 0.05 귀무 채택

print()
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkey = pairwise_tukeyhsd(df3.jikwon_pay, df3.buser_num)
print(turkey)
turkey.plot_simultaneous()






