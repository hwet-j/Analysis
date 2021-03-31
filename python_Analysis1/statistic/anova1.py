# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# '분산분석'이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.

# * 서로 독립인 세 집단의 평균 차이 검정
# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. three_sample.csv'
# 1개의 요인에 집단이 3개 : 일원분산분석(one-way anova)

# 귀무 : 교육생을 대상으로 세 가지 교육방법에 따른 실기시험 평균에 차이가 없다.
# 대립 : 교육생을 대상으로 세 가지 교육방법에 따른 실기시험 평균에 차이가 있다.

import pandas as pd
import scipy.stats as stats

data = pd.read_csv('../testdata/three_sample.csv')
print(data.head(3), data.shape)     # (80, 4)
print(data.describe())


import matplotlib.pyplot as plt
# plt.boxplot(data.score)
# plt.show()

data = data.query('score <= 100')
# plt.boxplot(data.score)
# plt.show()
print(data.describe())
# print(data)

# 독립성 : 상관관계를 확인 가능
result = data[['method','score']]
m1 = result[result['method'] == 1]
m2 = result[result['method'] == 2]
m3 = result[result['method'] == 3]
#print(m1)

score1 = m1['score']
score2 = m2['score']
score3 = m3['score']
print('등분산성 : ', stats.levene(score1, score2, score3).pvalue)   # 0.11322 > 0.05만족
print('등분산성 : ', stats.fligner(score1, score2, score3).pvalue)
print('등분산성 : ', stats.bartlett(score1, score2, score3).pvalue)


# 정규성 : 
print(stats.shapiro(score1))
print('정규성확인 : ', stats.ks_2samp(score1, score2))   # pvalue=0.309687 > 0.05 정규성 만족
print('정규성확인 : ', stats.ks_2samp(score1, score3))   # pvalue=0.716209
print('정규성확인 : ', stats.ks_2samp(score2, score3))   # pvalue=0.772408

print('----------------------')
print('교육방법별 건수')
data2 = pd.crosstab(index = data['method'], columns= 'count')
print(data2)

print('교육방법별 만족여부 건수')
data3 = pd.crosstab(data['method'], data['survey'])
data3.index = ['방법1', '방법2', '방법3']
data3.columns = ['만족', '불만족']
print(data3)

print('---------ANOVA------------')
import statsmodels.api as sm
from statsmodels.formula.api import ols
#model = ols('score ~ method', data).fit()
model = ols('score ~ C(method)', data).fit()    # C(변수명 + 변수명2 +......) ==>> 범주형임을 명시적으로 표시
table = sm.stats.anova_lm(model, typ=1)
print(table)    # p-value : 0.727597 > 0.05 이므로 귀무가설 채택. 교육생을 대상으로 세 가지 교육방법에 따른 실기시험 평균에 차이가 없다.
# print(model.summary())
# F 값은 mean_sq값의 method, survey값을 Residual값으로 나눈값...

print('---------ANOVA - 다중회귀 : 독립변수 2------------')
#model2 = ols('score ~ method + survey', data).fit()
model2 = ols('score ~ C(method + survey)', data).fit()  # C(변수명 + 변수명2 +......) ==>> 범주형임을 명시적으로 표시
table2 = sm.stats.anova_lm(model2, typ=1)
print(table2)

import numpy as np

print()
print(np.mean(score1))
print(np.mean(score2))
print(np.mean(score3))

# 사후 검정 : 그룹간에 평균값 차이가 의미가 있는지 확인 
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tResult = pairwise_tukeyhsd(data, data.method)
print(tResult)

tResult.plot_simultaneous()
plt.show()













