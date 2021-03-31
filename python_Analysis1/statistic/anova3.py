# 일원 분산분석
# 어느 음식점의 매출자료와 날씨 자료를 이용하여 온도에 따른 매출의 평균의 차이에 대한 검정
# 온도를 세그룹으로 분리

# 귀무 : 매출책 평균은 온도에 따라 차이가 없다
# 대립 : 매출책 평균은 온도에 따라 차이가 있다

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from pingouin.parametric import welch_anova

# 매출 자료
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3))   # 328
print(sales_data.info())

# 날씨 자료
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tweather.csv')
print(wt_data.head(3))  # 702
print(wt_data.info())

# 날짜를 기분으로 join(merge)
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3))
print()
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head, frame.shape)  # (328, 12)
print(frame.columns)

# 분석에 참여할 칼럼만 추출
data = frame.iloc[:, [0,1,7,8]]
print(data.head(3))

# 일별 최고온도를 구간설정을 통해 연속형 변수를 명목형 변수로 변경 작업
print(data.maxTa.describe())
# plt.boxplot(data.maxTa)
# plt.show()

# 온도를 추움, 보통, 더움 (0, 1, 2)으로 분류
data['Ta_gubun'] = pd.cut(data.maxTa, bins = [-5, 8, 24 , 37], labels = [0, 1, 2])
print(data.head(5))
print(data.isnull().sum())
# data = data[data.Ta_gubun.notna()]
# print(data.head(5))
print(data['Ta_gubun'].unique())

# 상관관계
print(data.corr())

# 세그룹으로 데이터를 나눈 후 등분산, 정규성 검정
x1 = np.array(data[data.Ta_gubun == 0].AMT)
x2 = np.array(data[data.Ta_gubun == 1].AMT)
x3 = np.array(data[data.Ta_gubun == 2].AMT)
print(x1)

print(stats.levene(x1, x2, x3))     # pvalue=0.039002 < 0.05 등분산성 만족X

print(stats.ks_2samp(x1, x2).pvalue, ' ', stats.ks_2samp(x1, x3).pvalue, ' ', stats.ks_2samp(x2, x3).pvalue)
# 9.28938415079017e-09   1.198570472122961e-28   1.4133139103478243e-13    < 0.05 정규성 만족X

# 온도별 매출액 평균
spp = data.loc[:, ['AMT', 'Ta_gubun']]
print(spp.groupby('Ta_gubun').mean())
print(pd.pivot_table(spp, index = ['Ta_gubun'], aggfunc = 'mean'))

# ANOVA 진행
sp = np.array(spp)
group1 = sp[sp[:, 1] == 0, 0]
group2 = sp[sp[:, 1] == 1, 0]
group3 = sp[sp[:, 1] == 2, 0]
print(group1)

print()
print(stats.f_oneway(group1, group2, group3))
# F_onewayResult(statistic=99.1908012029983, pvalue=2.360737101089604e-34)
# pvalue < 0.05  대립 : 매출책 평균은 온도에 따라 차이가 있다

print()
# 등분산성 만족X 경우 Welch's ANOVA ---> pip install pingouin
df = data
print(welch_anova(data = df, dv = 'AMT', between='Ta_gubun'))   # 7.907874e-35

print()
# 정규성 만족 X 경우 kruskal - wallis test
print(stats.kruskal(group1, group2, group3))
# KruskalResult(statistic=132.7022591443371, pvalue=1.5278142583114522e-29)
# 결론 : 온도에 따른 매출액의 차이가 유의미한 것으로 볼 수 있다.

# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
posthoc = pairwise_tukeyhsd(spp['AMT'], spp['Ta_gubun'])
print(posthoc)

posthoc.plot_simultaneous()
plt.show()

















