# T 검정 : 집단 간 평균(비율) 차이 검정
# t-test : 평균값의 차이와 표준편차의 비율이 얼마나 큰지 혹은 작은지를 통계적으로(과학적으로) 검정하는 방법
# 독립변수 : 범주,    종속변수 : 연속형

# 단일표본 검정(one-sample t-test) : 하나의 집단에 대한 표본평균이 에측된 평균과 같은지 여부를 검정
# 연습1) 어느 남성 집단의 평균키 검정
# 귀무 : 어느 남성 집단의 평균키가 177이다.
# 대립 : 어느 남성 집단의 평균키가 177이 아니다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

one_sample = [177.0, 182.7, 169.6, 176.8, 180.0]
print(np.array(one_sample).mean())

one_sample2 = [167.0, 162.7, 158.9, 173.9, 171.0]
print(np.array(one_sample2).mean())

result1 = stats.ttest_1samp(one_sample, popmean=177)  # sample의 개수가 하나일때
print('result1 : ', result1)    # pvalue=0.924864 > 0.05 귀무가설 채택 : 어느 남성집단의 평균키가 177이다.

result2 = stats.ttest_1samp(one_sample2, popmean=177)  # sample의 개수가 하나일때
print('result2 : ', result2)    # pvalue=0.019171 < 0.05 귀무가설 기각 : 어느 남성집단의 평균키가 177이 아니다.

result3 = stats.ttest_1samp(one_sample, popmean=167)
print('result3 : ', result3)    # pvalue=0.0095638 < 0.05 귀무가설 기각 : 어느 남성집단의 평균키가 167이 아니다.

print('\n=====================')
# 연습2) 어느 집단 자료 평균 검정
# 귀무 : 자료들의 평균은 0이다
# 대립 : 자료들의 평균은 0이 아니다

np.random.seed(123)
mu = 0
n = 10
x = stats.norm(mu).rvs(n)
print(x)
print(np.mean(x))


# x 데이터의 정규성 만족 여부 확인
#sns.distplot(x, kde=False, fit=stats.norm)  # 분포 시각화
#plt.show()
print(stats.shapiro(x))  # 정규성 확인 함수    pvalue=0.865896 > 0.05 정규성을 만족

result4 = stats.ttest_1samp(x, popmean=0)
print('result4 : ', result4)   # pvalue=0.529463 > 0.05 귀무 : 자료들의 평균은 0이다.  

# 참고 : 모수의 평균 0.8이라고 한다면
result5 = stats.ttest_1samp(x, popmean=0.8) 
print('result5 : ', result5)   # pvalue=0.0289619 < 0.05 귀무기각 : 자료들의 평균은 0.8이 아니다.  





