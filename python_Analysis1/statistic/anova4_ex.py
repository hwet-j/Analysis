# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

#귀무:기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하지 않는다.
#대립:기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재한다.

# 읽어온 데이터를 공백으로 해서 구분
data = pd.read_csv('anova_ex1.txt', sep=" ")
data.columns = ['종류','양']
print(data.head(6))
print(data.isnull().sum())   # 결측값 존재 여부 확인

data = data.fillna(data.mean())    # 평균으로 NaN값 대체
#data = data.fillna(data['양'].mean())  # 위와 동일
#print(data.head(6))

oil1 = data[data['종류'] == 1]
oil2 = data[data['종류'] == 2]
oil3 = data[data['종류'] == 3]
oil4 = data[data['종류'] == 4]
#print(oil1)
print(stats.ks_2samp(oil1['양'], oil2['양'])) # pvalue=0.9307359307359307
print("정규성 검정 : ",stats.shapiro(oil1['양'])) #0.8680403232574463  정규성 만족
print("정규성 검정 : ",stats.shapiro(oil2['양']))
print("정규성 검정 : ",stats.shapiro(oil3['양']))
print("정규성 검정 : ",stats.shapiro(oil4['양']))

print('등분산성 확인 :', stats.levene(oil1['양'], oil2['양'], oil3['양'], oil4['양']).pvalue) #0.32689 등분산성 만족
print()
print('방법 1')
f_sta, p_val = stats.f_oneway(oil1['양'], oil2['양'], oil3['양'], oil4['양'])
print("p-value : {}".format(p_val))    # p > 0.05이므로 귀무가설 채택 

print()
print('방법 2')
data = pd.DataFrame(data, columns=['종류','양'])
#print(data.head(3))
lmodel = ols('양 ~ C(종류)', data).fit()
print(anova_lm(lmodel))      # PR(>F) 0.848244  > 0.05이므로 귀무가설 채택

print('--------------------------------------')
#[ANOVA 예제 2]
#DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.
#귀무 : 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 없다.
#대립 : 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있다.

import MySQLdb  
from pandas import DataFrame
import ast

try:
    with open('mariadb.txt','r') as f:
        config = f.read()
except Exception as e:
    print('read err: ', e)

config = ast.literal_eval(config)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = "select buser_num, jikwon_pay from jikwon"
    cursor.execute(sql)
    jikdata = DataFrame.from_records(cursor.fetchall(), columns=['부서번호','연봉'])
    #print(jikdata.head(5))
except Exception as e:
    print("err : ",e)
finally: 
    cursor.close()
    conn.close()

buser10 = jikdata[jikdata['부서번호'] == 10]
buser20 = jikdata[jikdata['부서번호'] == 20]
buser30 = jikdata[jikdata['부서번호'] == 30]
buser40 = jikdata[jikdata['부서번호'] == 40]
print(buser10.head(4))

print("정규성 검정 : ", stats.shapiro(buser10['연봉'])) #0.0260
print("정규성 검정 : ", stats.shapiro(buser20['연봉'])) 
print("정규성 검정 : ", stats.shapiro(buser30['연봉'])) 
print("정규성 검정 : ", stats.shapiro(buser40['연봉']))  

print('등분산성 확인 :', stats.bartlett(buser10['연봉'], buser20['연봉'], buser30['연봉'], buser40['연봉']).pvalue) #0.627 등분산성 만족
print()

print('방법 1')
f_sta, p_val = stats.f_oneway(buser10['연봉'], buser20['연봉'], buser30['연봉'], buser40['연봉'])
print("f_sta : {}".format(f_sta))
print("p-value : {}".format(p_val))    # p-value > 0.05이므로 귀무가설 채택

print('방법 2')
jikdata = pd.DataFrame(jikdata, columns=['부서번호','연봉'])
# print(jikdata.head(3))
lmodel = ols('연봉 ~ C(부서번호)', jikdata).fit()
print(anova_lm(lmodel))   # PR(>F) 0.740797
