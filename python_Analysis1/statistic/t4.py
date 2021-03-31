# 어느 음식점의 매출자료와 날씨 자료를 이용하여 강수여부에 따른 매출의 평균의 차이에 대한 검정
# 집단1 : 비가 올때의 매출, 집단2 : 비가 안올때의 매출
# 강수여부에 따른 매출액의 평균의 차이가 없다
# 강수여부에 따른 매출액의 평균의 차이가 있다

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

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

print()
#print(data['sumRn'] > 0)
# 방법 1 - True, False를 문법으로 교체 (.astype(int))
#data['rain_yn'] = (data['sumRn'] > 0).astype(int)   # 0보다 크면 True 아니면 False(astype(int)를 써줌으로써 True는 1로 False는 0으로
#print(data.head(3))

# 방법 2 - True, False를  *1을 사용해 교체
#print(True * 1, False * 1)
data['rain_yn'] = (data.loc[:, ('sumRn')] > 0) * 1
print(data.head(3))

# 비 여부에 따른 매출액 비교용 시각화
sp = np.array(data.iloc[:, [1, 4]])   # AMT, rain_yn을 출력
tg1 = sp[sp[:, 1] == 0, 0]  # 비 안올때 매출액
tg2 = sp[sp[:, 1] == 1, 0]  # 비 올때 매출액
print(tg1[:3])
print(tg2[:3])
print(np.mean(tg1), np.mean(tg2))   # tg1 : 761040.254237, tg2 : 757331.521739

# plt.plot(tg1)
# plt.show()
# plt.plot(tg2)
# plt.show()

# plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
# plt.show()

# 두 집단 평균차이 검정
# 정규성 (N > 30)
print(len(tg1), ' ', len(tg2))
print(stats.shapiro(tg1).pvalue)    # 0.05604 > 0.05 만족
print(stats.shapiro(tg2).pvalue)    # 0.88273

# 등분산 
print(stats.levene(tg1, tg2).pvalue)    # 0.71234

print()
print(stats.ttest_ind(tg1, tg2, equal_var = True))
# Ttest_indResult(statistic=0.10109828602924716, pvalue=0.919534587722196)
# 결론  : pvalue=0.91953 > 0.05 귀무가설 채택
# 강수여부에 따른 매출액 평균에 차이가 없다.









