# 서로 대응인 두 집단의 평균 차이 검정(paired samples t-test)
# 처리 이전과 처리 이후를 각각의 모집단으로 판단하여, 동일한 관찰 대상으로부터 처리 이전과 처리 이후를 1:1로 대응시킨 두 집단으로 부터
# 의 표본을 대응표본(paired sample)이라고 한다.
# 대응인 두 집단의 평균 비교는 동일한 관찰 대상으로부터 처리 이전의 관찰과 이후의 관찰을 비교하여 영향을 미친 정도를 밝히는데 주로 사용
# 하고 있다. 집단 간 비교가 아니므로 등분산 검정을 할 필요가 없다

# 광고 전후의 상품 판매량의 차이, 운동 전후의 근육량의 차이...

import numpy as np
import scipy as sp
import scipy.stats as stats

# 대응표본 t검정 연습1
# 귀무 : 특강 전후의 시험점수는 차이가 없다.
# 대립 : 특강 전후의 시험점수는 차이가 있다.
np.random.seed(12)
x1 = np.random.normal(80, 10, 100)
x2 = np.random.normal(77, 10, 100)
print('x1 : ',x1)

# 정규성
import matplotlib.pyplot as plt
import seaborn as sns
# sns.distplot(x1, kde = False, fit = stats.norm)
# sns.distplot(x2, kde = False, fit = stats.norm)
# plt.show()
print(stats.shapiro(x1))    # pvalue=0.99421 > 0.05 정규성 만족
print(stats.shapiro(x2))    # pvalue=0.79854
# 집단이 하나이므로 등분산성 검정 X
print(stats.ttest_rel(x1, x2))
# Ttest_relResult(statistic=2.388932926547383, pvalue=0.018792247355705678)
# pvalue=0.01879 < 0.05 이므로 귀무가설 기각
# 대립 : 특강 전후의 시험점수는 차이가 있다.

#-----------------------------------------
print('---------------------------')
# 실습) 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게 변화
# 귀무 : 복부 수술 전 몸무게와 복부 수술 후 몸무게의 변화가 없다.
# 대립 : 복부 수술 전 몸무게와 복부 수술 후 몸무게의 변화가 있다.

baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]
print(np.mean(baseline))    # 78.41
print(np.mean(follow_up))   # 71.5
print(np.mean(baseline) - np.mean(follow_up))   # 6.911

result = stats.ttest_rel(baseline, follow_up)
print(result)
# Ttest_relResult(statistic=3.6681166519351103, pvalue=0.006326650855933662)
# pvalue=0.006326 < 0.05이므로 귀무가설 기각. 복부 수술전 몸무게와 복부 수술후 몸무게의 변화가 있다.














