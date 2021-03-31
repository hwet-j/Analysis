# 회귀분석 문제 3) 
# kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수와 선형회귀모델 충족 조건(선형성, 독립성, 정규성, 등분산성, 다중공선성 등)을 활용하여
# 타당한 변수만 임의적으로 선택한다. 완성된 모델로 Sales를 예측.

# https://ysyblog.tistory.com/120

import scipy.stats as stats
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')


data = pd.read_csv('Carseats.csv')
print(data.head(3), ' ', data.shape)  # (400, 11)
# print(data.index, data.columns)
# print(data.info())

print()
# print('상관계수(r) : ', data.loc[:, ['Sales','CompPrice']].corr())    # 0.064079
# 거의 무시될수 있는  상관관계 


# print('상관계수(r) : ', data.loc[:, ['Sales','Income']].corr())    # 0.151951
# 약한 양의 상관관계
lm = smf.ols(formula = 'Sales ~ Income', data = data).fit()
# 예측값
pred = lm.predict(data)
# 잔차
residual = data['Sales'] - pred
# print(residual)
# 선형성
# sns.regplot(pred, residual, lowess=True, line_kws={'color':'red'})
#plt.plot([pred.min(), pred.max()], [0,0],'--', color='grey')
# plt.show()
# 선형성이 있음 - Income
# 빨간실선은 잔차의 추세를 나타냄 , 빨간실선이 점선을 크게 벗어나면 예측값에 따라 잔차가 크게 달라진다는 것으로 선형성이 없다는것.

# 잔차의 정규성
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
# sns.scatterplot(x, y)
# plt.plot([-3, 3], [-3, 3], '--', color='grey')
# plt.show()
print(stats.shapiro(residual))
# pvalue=0.26704 > 0.05 이므로 잔차 정규성만족 - Income

# 잔차의 등분산성
# sns.regplot(pred, np.sqrt(np.abs(sr)), lowess=True, line_kws={'color':'red'})
# plt.show()
# 빨간색 실선이 수평선에 가까울수록 등분산성이 있다는것.
# 등분산성이 있다고 판단. - Income

# 독립성
# Durbin-Watson: 잔차항이 독립성을 만족하는지 확인 가능. 2에 가까우면 자기상관이 없다.(서로 독립 - 잔차끼리 상관관계가 없다.)
# 0에 가까우면 양의 상관, 4에 가까우면 음의 상관, 보통 1.5~2.5사이면 독립으로 판단하고 회귀모형이 적합하다고 생각함.
# print(lm.summary())
# Durbin-Watson: 1.864 - 독립성 존재


# print('상관계수(r) : ', data.loc[:, ['Sales','Advertising']].corr())    # 0.269507
# 약한 양의 상관관계
lm2 = smf.ols(formula = 'Sales ~ Advertising', data = data).fit()
# 예측값
pred2 = lm2.predict(data)
# 잔차
residual2 = data['Sales'] - pred2
# print(residual2)
# sns.regplot(pred2, residual, lowess=True, line_kws={'color':'red'})
# plt.plot([pred2.min(), pred2.max()], [0,0],'--', color='grey')
# plt.show()
# 선형성이 있음 - Advertising

# 잔차의 정규성
sr2 = stats.zscore(residual2)
(x, y), _ = stats.probplot(sr2)
# sns.scatterplot(x, y)
# plt.plot([-3, 3], [-3, 3], '--', color='grey')
# plt.show()
print(stats.shapiro(residual2))
# pvalue=0.202427 > 0.05 이므로 잔차 정규성만족



# print('상관계수(r) : ', data.loc[:, ['Sales','Population']].corr())    # 0.050471
# 거의 무시될수 있는  상관관계




# print('상관계수(r) : ', data.loc[:, ['Sales','Price']].corr())    # -0.444951
# 뚜렷한 음의 선형관계
lm3 = smf.ols(formula = 'Sales ~ Price', data = data).fit()
# 예측값
pred3 = lm3.predict(data)
# 잔차
residual3 = data['Sales'] - pred3
# print(residual3)
# sns.regplot(pred3, residual, lowess=True, line_kws={'color':'red'})
# plt.plot([pred3.min(), pred3.max()], [0,0],'--', color='grey')
# plt.show()
# 선형성이 없음? - Price

# 잔차의 정규성
sr3 = stats.zscore(residual3)
(x, y), _ = stats.probplot(sr3)
# sns.scatterplot(x, y)
# plt.plot([-3, 3], [-3, 3], '--', color='grey')
# plt.show()
print(stats.shapiro(residual3))
# pvalue=0.2700171 > 0.05 이므로 잔차 정규성만족


# print('상관계수(r) : ', data.loc[:, ['Sales','Age']].corr())    # -0.231815
# 약한 음의 선형관계
lm4 = smf.ols(formula = 'Sales ~ Age', data = data).fit()
# 예측값
pred4 = lm4.predict(data)
# 잔차
residual4 = data['Sales'] - pred4
# print(residual4)
# sns.regplot(pred4, residual, lowess=True, line_kws={'color':'red'})
# plt.plot([pred4.min(), pred4.max()], [0,0],'--', color='grey')
# plt.show()
# 선형성 ?????????? - Age

# 잔차의 정규성
sr4 = stats.zscore(residual4)
(x, y), _ = stats.probplot(sr4)
# sns.scatterplot(x, y)
# plt.plot([-3, 3], [-3, 3], '--', color='grey')
# plt.show()
print(stats.shapiro(residual4))
# pvalue=0.3037887 > 0.05 이므로 잔차 정규성만족



# print('상관계수(r) : ', data.loc[:, ['Sales','Education']].corr())    # -0.051955
# 거의 무시될수 있는  상관관계




# 피어슨 상관계수
# -1 < r < 0.7    : 강한 음의 선형관계
# -0.7 < r < -0.3 : 뚜렷한 음의 선형관계
# -0.3 < r < -0.1 : 약한 음의 선형관계
# -0.1 < r < 0.1  : 거의 무시될 수 있는 선형관계
# 0.1 < r < 0.3   : 약한 양의 상관관계
# 0.3 < r < 0.7   : 뚜렷한 양의 선형관계
# 0.7 < r < 1     : 강한 양의 선형관계





