# 선형회귀분석 : 여러 매체의 광고비에 따른 판매량. ols(). 모델 작성후 추정치 얻기
import scipy.stats as stats
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

adfdf = pd.read_csv("../testdata/Advertising.csv", usecols=[1,2,3,4])
print(adfdf.head(3), ' ', adfdf.shape)  # (200, 4)
print(adfdf.index, adfdf.columns)
print(adfdf.info())

print()
print('상관계수(r) : ', adfdf.loc[:, ['sales','tv']].corr())    # 0.782224
# 강한 양의 상관관계이고, 인과관계임을 알 수 있다.
# lm = smf.ols(formula = 'sales ~ tv', data = adfdf)
# lm_learn = lm.fit()
lm = smf.ols(formula = 'sales ~ tv', data = adfdf).fit()
#print(lm.summary())
print(lm.params)
print(lm.pvalues)
print(lm.rsquared)

# 시각화
'''
import seaborn as sns
plt.scatter(adfdf.tv, adfdf.sales)
plt.xlabel('tv')
plt.ylabel('sales')
x = pd.DataFrame({'tv':[adfdf.tv.min(), adfdf.tv.max()]})
y_pred = lm.predict(x)
plt.plot(x, y_pred, c='red')
plt.title('Linear Regression')
sns.regplot(adfdf.tv, adfdf.sales, scatter_kws = {'color':'r'})
plt.xlim(-50, 350)
plt.ylim(ymin=0)
plt.show()
'''

# 예측 : 새로운 tv 값으로 sales를 추정
x_new = pd.DataFrame({'tv':[230.1, 44.5, 100]})
print('x_new', x_new)
pred = lm.predict(x_new)
print('추정값 : ', pred)

print('\n다중 선형회귀 모델------------')
lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data = adfdf).fit()
print(lm_mul.summary())

print(adfdf.corr())

# 예측2 : 새로운 tv, radio 값으로 sales를 추정
x_new2 = pd.DataFrame({'tv':[230.1, 44.5, 100], 'radio':[30.1, 40.1, 50.1],\
                      'newspaper':[10.1, 10.1, 10.1]})
pred2 = lm.predict(x_new)
print('추정값 : ', pred2)

# 회귀분석모형의 적절성을 위한 조건 : 아래의 조건 위배 시에는 변수 제거나 조정을 신중히 고려해야 함.
# - 정규성 : 독립변수들의 잔차항이 정규분포를 따라야 한다.
# - 독립성 : 독립변수들 간의 값이 서로 관련성이 없어야 한다.
# - 선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지면 좋지 않다.
# - 등분산성 : 독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.
# - 다중공선성 : 독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.

# 잔차항
fitted = lm_mul.predict(adfdf)  # 예측값
# print(fitted)
residual = adfdf['sales'] - fitted  # 잔차

import seaborn as sns
print('선형성 - 예측값과 잔차가 비슷하게 유지 --------')
'''
sns.regplot(fitted, residual, lowess = True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='grey')
plt.show()  # 선형성을 만족하지 못함
'''

print('정규성 - 잔차가 정규분포를 따르는지 확인 --------')
import scipy.stats
sr = scipy.stats.zscore(residual)
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3], [-3, 3], '--', color='grey')
plt.show()

print('shapiro test :', scipy.stats.shapiro(residual))
# shapiro test : ShapiroResult(statistic=0.9176644086837769, pvalue=3.938041004403203e-09)
# pvalue=3.938041004403203e-09 < 0.05 이므로 정규성 만족하지 못함

print('\n독립성 - 잔차가 자기상관(인접 관측치의 오차가 상관되어 있음)이 있는지 확인 --------')
# 모델.summary() 하면 Durbin-Watson:2.084 
# Durbin-Watson: 잔차항이 독립성을 만족하는지 확인 가능. 2에 가까우면 자기상관이 없다.(서로 독립 - 잔차끼리 상관관계가 없다.)
# 0에 가까우면 양의 상관, 4에 가까우면 음의 상관

print('\n등분산성 - 잔차의 분산이 일정한지 확인 --------')
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess = True, line_kws={'color':'red'})
plt.show()
# 빨간색 실선이 수평선을 그리지 않으므로 n등분산성을 만족하지 못함.

print('\n다중공선성 (Multicollinearity) - 독립변수들 간에 강한 상관관계 확인 --------')
# VIF(Variance Inflation Factors, 분산 인플레 요인) 값이 10을 넘으면 다중공선성이 발생하는 변수라고 할 수 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(adfdf.values, 0))
print(variance_inflation_factor(adfdf.values, 1))
print(variance_inflation_factor(adfdf.values, 2))
print(variance_inflation_factor(adfdf.values, 3))

# DataFrame으로 보기
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(adfdf.values, i) for i in range(adfdf.shape[1])]
print(vifdf)

print("\n참고 : Cook's distance - 극단값을 나타내는 지표 확인 ---------")
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm_mul).cooks_distance
print(cd.sort_values(ascending=False).head())

import statsmodels.api as sm
sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()

print(adfdf.iloc[[130, 5, 75, 35, 178]])    # 이 값들은 작업에서 제외하기를 권장






