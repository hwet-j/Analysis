'''
회귀분석 문제 3) 
kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
변수 선택은 모델.summary() 함수와 선형회귀모델 충족 조건(선형성, 독립성, 정규성, 등분산성, 다중공선성 등)을 활용하여
타당한 변수만 임의적으로 선택한다. 완성된 모델로 Sales를 예측.
ols
LinearRegression
'''
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('carseats.csv')
#print(df.columns)
df = df.drop([df.columns[6], df.columns[9], df.columns[10]], axis=1)    # 범주형 제외
print(df.head(3))
print(df.columns)
# 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education'
#print(df.corr())
#plt.scatter(df.CompPrice, df.Sales) # 0.064079
#plt.scatter(df.Income, df.Sales)    # 0.151951
#print(np.corrcoef(df.Advertising, df.Sales)) # 0.26950678
#print(np.corrcoef(df.Population, df.Sales))  # 0.05047098
#plt.scatter(df.Advertising, df.Sales)
#plt.scatter(df.Population, df.Sales)
#plt.scatter(df.Price, df.Sales) # -0.444951
#plt.scatter(df.Age, df.Sales) # -0.231815
#plt.show()

#print(df.info())
#model = smf.ols(formula='Sales ~ Age + Price', data = df).fit()
#model = smf.ols(formula='Sales ~ Price + CompPrice', data = df).fit()
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#model = smf.ols(formula='Sales ~ Price + Income + Advertising  + Age', data = df).fit()
#model = smf.ols(formula='Sales ~ Price + Income + Advertising  + Age', data = df).fit()
model = smf.ols(formula='Sales ~ Price + Income + Advertising  + Age', data = df).fit()
# Education 변수는 p값이 0.468 > 0.05 이므로 모델의 유의성 향상을 위해 제외
# Comp 변수는 다중 공선성이 127.737980으로 제외
# shapiro pvalue = 0.2127407 > 0.05으로 정규성을 만족
# R-squared : 0.364 정도의 설명력을 가진 모델 선정



print(model.params)
print('pvalues : ', model.pvalues)
print('rsquared : ', model.rsquared)
# R-squared:                       0.371
print(model.summary())


# 잔차항
fitted = model.predict(df)     # 예측값
print(fitted)
residual = df['Sales'] - fitted # 잔차

# 선형성
sns.regplot(fitted, residual, lowess = True, line_kws = {'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
plt.show() 

# 정규성
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3], [-3, 3], '--', color="grey")
plt.show() # 선형성을 만족하지 못한다. 
print('residual test :', stats.shapiro(residual))
# pvalue=0.2127407342195511

# 독립성
#Durbin-Watson:                   1.931

# 등분산성
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess = True, line_kws = {'color':'red'})
plt.show()

# 다중 공선성
vif_df = pd.DataFrame()
vif_df['vid_value'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print(vif_df)