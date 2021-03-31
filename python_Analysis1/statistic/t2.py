# 단일표본 검정(one-sample t-test)

# 실습 예제 1)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리(국어 점수 80점에 대한 평균검정) student.csv
# 귀무 : 학생들의 국어점수 평균은 80이다.
# 대립 : 학생들의 국어점수 평균은 80점이 아니다

import pandas as pd
import scipy.stats as stats
import numpy as np

data = pd.read_csv("../testdata/student.csv")
print(data.head(3))
print(data.describe())
print(np.mean(data.국어)) # 72.9와 80은 평균에 차이가 있는가?

result = stats.ttest_1samp(data.국어, popmean=80)
print('result : ', result)  #  pvalue=0.198560 > 0.05 귀무가설 채택
# 데이터에따르면 72.9와 80점은 차이가 없다고 할수있다.

# 참고 
result2 = stats.ttest_1samp(data.국어, popmean=60)
print('result2 : ', result2)  #  pvalue=0.025687 < 0.05 귀무가설 기각
# 데이터에따르면 72.9와 60점은 차이가 있다고 할수있다.


# 실습 예제 2)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

# 귀무 : 여아 신생아의 몸무게는 평균이 2800(g)이다.
# 대립 : 여아 신생아의 몸무게는 평균이 2800(g)이 아니다.

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/babyboom.csv")
# gender 1 여자, 2 남자
print(data.head(3))
print(data.describe())

fdata = data[data.gender == 1]  # gender 1값을 따로 뽑아냄
print(fdata, len(fdata), np.mean(fdata.weight))    # 18명의 몸무게평균 :3132.444444(g)

# 통계 추론시 대부분의 모집단은 정규분포를 따른다는 가정하에 진행하는 것이 일반적
# - 중심극한의 원리 (표본의 크기가 커질수록 표본 평균의 분포는 모집단의 분포 모양과는 관계없이 정규 분포에 가까워 진다.)의 이해
# 정규성 확인
print(stats.shapiro(fdata['weight']))   # pvalue=0.01798479 < 0.05 정규성 위반 

import matplotlib.pyplot as plt
import seaborn as sns
#sns.distplot(fdata.iloc[:,2], fit=stats.norm)
#plt.show()

#stats.probplot(fdata.iloc[:,2], plot=plt)   # Q-Q plot 상에서 정규성 확인
#plt.show()

result3 = stats.ttest_1samp(fdata.weight, popmean=2800)
print('result3 : ', result3)    
# pvalue=0.0392684 < 0.05 이므로 귀무 기각 
# 결론 = 대립 : 여아 신생아 몸무게는 평균이 2800이 아니다.