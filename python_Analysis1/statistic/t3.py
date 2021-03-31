# 차이분석 중 두 집단 평균차이 검정
# 선행조건 : 1. 두 집단은 정규분포를 따라야 한다. 
#         2. 두 집단의 분산이 동일해야 한다.
import numpy as np
from scipy import stats
import pandas as pd

# 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.

# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
# 귀무 : 남녀 두 집단 간의 시험 평균이 차이가 없다.
# 대립 : 남녀 두 집단 간의 시험 평균이 차이가 있다.

# 데이터가 30개 이상이면 정규성을 따르고, 그 밑이면 따르지 않는다.(중심극한정리 이론에 의해)
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
print(np.mean(male))    # 83.8
print(np.mean(female))  # 72.24

two_sample = stats.ttest_ind(male, female)    # 등분산을 만족한 경우
# two_sample = stats.ttest_ind(male, female, equal_var = True)    # 위와 동일
print(two_sample)   # statistic=1.233193127514512, pvalue=0.25250768448532773
print(two_sample.pvalue)    # pvalue=0.25250768448 > 0.05   귀무 채택

print('\n=======================')
# 실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'
# 귀무 : 두 가지 교육방법에 따른 시험 평균이 차이가 없다.
# 대립 : 두 가지 교육방법에 따른 시험 평균이 차이가 있다.

data = pd.read_csv('../testdata/two_sample.csv')
print(data)
ms = data[['method','score']]   # 교육 방법과 점수만 따로 추출
print(ms)
# 교육방법 별로 데이터 분리
m1 = ms[ms['method'] == 1]  # 교육방법 1
m2 = ms[ms['method'] == 2]  # 교육방법 2
print(m1[:5])
print(m2[:5])

sco1 = m1['score']  # 교육방법1의 점수
sco2 = m2['score']  # 교육방법2의 점수

# NaN - 임의의 값으로 대체, 평균으로 대체, 제거 
# sco1 = sco1.fillna(0)    # 0으로 채워넣음
sco1 = sco1.fillna(sco1.mean()) # 평균으로 대체
sco2 = sco2.fillna(sco2.mean())
print()
print(sco1[:5])
print(sco2[:5])

# 정규성 확인
import matplotlib.pyplot as plt
import seaborn as sns
# sns.distplot(sco1, kde = False, fit=stats.norm)
# sns.distplot(sco2, kde = False, fit=stats.norm)
# plt.show()

print(stats.shapiro(sco1))      # pvalue=0.3679903 > 0.05 정규성 만족
print(stats.shapiro(sco2))      # pvalue=0.6714189

# 등분산성
# bartlett : scipy.stats.bartlett
# fligner : scipy.stats.fligner
# levene : scipy.stats.levene
# 비모수(데이터가 30개 이하)일일때 bartlett 사용
print(stats.bartlett(sco1, sco2))   # pvalue=0.3679903 > 0.05 등분산성 만족
print(stats.fligner(sco1, sco2))    # pvalue=0.6714189
# 일반적으로 levene을 사용함 
print(stats.levene(sco1, sco2))     # pvalue=0.4568427
#print(stats.levene(sco1, sco2).pvalue)     # pvalue값만 출력

result = stats.ttest_ind(sco1, sco2, equal_var = True)  # 정규성 만족, 등분산성 만족
print(result)
# Ttest_indResult(statistic=-0.19649386929539883, pvalue=0.8450532207209545)
# 결론 :  pvalue=0.845053 > 0.05 귀무채택

# 참고 - 크기가 안맞아 오류남
# result1 = stats.ttest_ind(sco1, sco2, equal_var = False)    # 정규성 만족, 등분산성 만족 X
# result1 = stats.wilcoxon(sco1, sco2)    # 정규성 만족 X







