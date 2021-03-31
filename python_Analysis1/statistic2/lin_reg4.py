
# 분석을 할거라면 탠서플로우?도 대충 할줄알아야함..

# 단순선형회귀 : ols()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3))

# 피어슨 상관계수
print(df.corr())    # 적절성/만족도의 상관관계 0.766853

print('-------------')
import statsmodels.formula.api as smf
model = smf.ols(formula = '만족도 ~ 적절성', data=df).fit()
print(model.summary())  # 이 값들에 따라 무슨의미를 뜻하는지 알아보면 좋음 
print( 0.766853 ** 2)   # 0.588063523609
# R-squared(결정계수, 설명력):0.588     상관계수 r을 제곱 or 1 - (SSE / SST)
# 설명력은 1에가까울수록 좋은 모델

print(model.params)     # 절편, 기울기
print(model.rsquared)   # 결정계수
print(model.pvalues)    # 결정계수
# print(model.predict())
print(df.만족도[0], ' ', model.predict()[0])   # 3    3.73

# 새로운 값 예측
print(df.적절성[:5])
print(df.만족도[:5])
print(model.predict()[:5])
print()

new_df = pd.DataFrame({'적절성':[6,5,4,3,22]})
new_pred = model.predict(new_df)
print('new_pred :',new_pred)

# 시각화
plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)    # R의 abline 기능
plt.plot(df.적절성, df.적절성 * slope + intercept, 'b')   # 추세선 그림
plt.show()

print('독립변수가 복수 : 다중선형회귀------')
model2 = smf.ols(formula = '만족도 ~ 적절성 + 친밀도', data=df).fit()
print(model2.summary())



