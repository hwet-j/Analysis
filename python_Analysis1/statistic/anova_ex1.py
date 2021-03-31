# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
import scipy.stats as stats
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 귀무 : 네 가지 기름에 따른 빵 흡수량에 차이가 없다.
# 대립 : 네 가지 기름에 따른 빵 흡수량에 차이가 있다.

data = pd.read_csv('anova_ex1.txt',delimiter=' ')
data['quantity'] = data['quantity'].fillna(data['quantity'].mean()) # 평균으로 대체
#print(data)

# 네 개의 집단
m1 = data[data['kind'] == 1]
m2 = data[data['kind'] == 2]
m3 = data[data['kind'] == 3]
m4 = data[data['kind'] == 4]

gr1 = m1['quantity']
gr2 = m2['quantity']
gr3 = m3['quantity']
gr4 = m4['quantity']

#print(gr1, np.average(gr1))     # 63.25438596491228
#print(gr2, np.average(gr2))     # 68.833333
#print(gr3, np.average(gr3))     # 66.75
#print(gr4, np.average(gr4))     # 72.75

# 정규성
#print(stats.shapiro(gr1))   # pvalue=0.868040
#print(stats.shapiro(gr2))   # pvalue=0.592393
#print(stats.shapiro(gr3))   # pvalue=0.486010
#print(stats.shapiro(gr4))   # pvalue=0.416217
print('정규성확인 : ', stats.ks_2samp(gr1, gr2))   # pvalue=0.930735 > 0.05 정규성 만족
print('정규성확인 : ', stats.ks_2samp(gr1, gr3))   # pvalue=0.923809
print('정규성확인 : ', stats.ks_2samp(gr1, gr4))   # pvalue=0.552380
print('정규성확인 : ', stats.ks_2samp(gr2, gr3))   # pvalue=0.923809
print('정규성확인 : ', stats.ks_2samp(gr2, gr4))   # pvalue=0.552380
print('정규성확인 : ', stats.ks_2samp(gr3, gr4))   # pvalue=0.771428

print('----------------------')
print('기름별 튀긴횟수')
data2 = pd.crosstab(index = data['kind'], columns= 'count')
print(data2)

print('---------ANOVA------------')

model = ols('quantity ~ C(kind)', data).fit()    # C(변수명 + 변수명2 +......) ==>> 범주형임을 명시적으로 표시
table = sm.stats.anova_lm(model, typ=1)
print(table)    # p-value : 0.848244 > 0.05 이므로 귀무 : 네 가지 기름에 따른 빵 흡수량에 차이가 없다.




