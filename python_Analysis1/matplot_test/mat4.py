import pandas as pd

tips = pd.read_csv("../testdata/tips.csv")
print(tips.info())
print(tips.head(3))

tips['gender'] = tips['sex']    # sex와 동일한 칼럼 gender칼럼을 생성
del tips['sex'] # sex칼럼을 제거
print(tips.head(3))

# 팁 비율 칼럼 추가 
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(3))

print()
tip_pct_group = tips['tip_pct'].groupby([tips['gender'], tips['smoker']]) # 성별, 흡연자별 그룹화
print(tip_pct_group)    # SeriesGroupBy object
print(tip_pct_group.sum())
print(tip_pct_group.max())
print(tip_pct_group.min())

result = tip_pct_group.describe()
print(result)

print('\n\n')
print(tip_pct_group.agg('sum'))
print(tip_pct_group.agg('mean'))
print(tip_pct_group.agg('max'))
print(tip_pct_group.agg('min'))

def diffFunc(group):
    diff = group.max() - group.min()
    return diff

result2 = tip_pct_group.agg(['var','mean','max', diffFunc])
print(result2)

import matplotlib.pyplot as plt
result2.plot(kind = 'barh', title='aff fund', stacked = True)
plt.show()





