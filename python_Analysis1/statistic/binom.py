# 이항 검정 : 결과가 두가지 값을 가지는 확률변수의 분포를 판단하는데 효과적
# 이산변량을 대상으로 한다.

# 형식 : stats.binom_test() : 명목척도의 비율을 바탕으로 이항분포 검정

import pandas as pd
import scipy.stats as stats

data = pd.read_csv("../testdata/one_sample.csv")
print(data.head(3))

# 귀무 : 직원을 대상으로 고객대응 교육 후 고객 안내 서비스 만족율이 80%다.
# 대립 : 직원을 대상으로 고객대응 교육 후 고객 안내 서비스 만족율이 80%이 아니다.

ctab = pd.crosstab(index = data['survey'], columns= 'count')
ctab.index = ['불만족', '만족']
print(ctab)

print('양측검정(기존 80% 만족율 기준 검증율 실시): 방향성이 없다.')
x = stats.binom_test([136, 14], p = 0.8, alternative = 'two-sided')
print(x)    # 0.00067347 < 0.05 이므로 귀무가설 기각. 
# 양측검정에서는 "직원을 대상으로 고객대응 교육 후 고객 안내 서비스 만족율이 80%가 아니다."로 표현 크다,작다 방향성을 제시하지 않는다.

print('양측검정(기존 20% 불만족율 기준 검증율 실시): 방향성이 없다.')
x = stats.binom_test([14, 136], p = 0.2, alternative = 'two-sided')
print(x)    # 0.00067347 < 0.05 이므로 귀무가설 기각.

print('\n단측검정(기존 80% 만족율 기준 검증율 실시): 방향성이 있다.')
# greater 만족율이 더 클것이다를 가정으로 사용
x = stats.binom_test([136, 14], p = 0.8, alternative = 'greater')
print(x)    # 0.000317940 < 0.05 이므로 귀무가설 기각.
  
print('단측검정(기존 20% 불만족율 기준 검증율 실시): 방향성이 있다.')
# less 불만족율이 더 작을것이다를 가정으로 사용
x = stats.binom_test([14, 136], p = 0.2, alternative = 'less')
print(x)    # 0.000317940 < 0.05 이므로 귀무가설 기각.

print('\n---------------------------')
# 비율 검정 : 집단의 비율이 어떤 특정한 값과 같읕지를 검정

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# one-sample
# a회사에는 100명 중 45명이 흡연을 한다. 국가통계를 보니 국민 흡연율은 35%라고 한다.
# 귀무 : a회사의 흡연율과 국민 흡연율은 비율이 같다.
# 대립: a회사의 흡연율과 국민 흡연율은 비율이 다르다.

count = np.array([45])
nobs = np.array([100])
val = 0.35

z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)
print(p)    # p-value 0.04442 < 0.05  귀무 기각. a회사의 흡연율과 국민 흡연율은 비율이 다르다.


# two-sample
# a회사에는 300명 중 100명이 햄버거를 먹었고, b회사 직원 400명 중 170명이 햄버거를 먹었다고 할 때
# 두 집단의 햄버거를 먹은 비율의 차이 검정
# 귀무 : 차이가 없다.
# 대립 : 차이가 있다.

count = np.array([100, 170])
nobs = np.array([300, 400])

z, p = proportions_ztest(count=count, nobs=nobs, value=0)
print(z)
print(p)    # p-value 0.013675 < 0.05 귀무 기각












