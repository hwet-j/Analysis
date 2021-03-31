# 대표적인 분류/예측 모델로 Regression 연습
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

adver = pd.read_csv("../testdata/Advertising.csv", usecols=[1,2,3,4])
print(adver.head(3))

x = np.array(adver.loc[:,'tv':'newspaper'])
y = np.array(adver.sales)
print(x[:2])
print(y[:2])

print('KNeighborsRegressor')
kmodel = KNeighborsRegressor(n_neighbors = 3).fit(x, y)
print(kmodel)
kpred = kmodel.predict(x)
print('kpred : ', kpred[:5])
print('k_r2 : ', r2_score(y, kpred))    # 0.96801

# --------------------------------
print('LinearRegression')
lmodel = LinearRegression().fit(x, y)
print(lmodel)
lpred = lmodel.predict(x)
print('lpred : ', lpred[:5])
print('l_r2 : ', r2_score(y, lpred))    # 0.8972106


# --------------------------------
print('RandomForestRegressor')
rmodel = RandomForestRegressor(n_estimators = 100, criterion = 'mse').fit(x, y)
print(rmodel)
rpred = rmodel.predict(x)
print('rpred : ', rpred[:5])
print('r_r2 : ', r2_score(y, rpred))    # 0.9975000


# --------------------------------
print('XGBRegressor')
xmodel = XGBRegressor(n_estimators = 100).fit(x, y)
print(xmodel)
xpred = xmodel.predict(x)
print('xpred : ', xpred[:5])
print('x_r2 : ', r2_score(y, xpred))    # 0.9999996