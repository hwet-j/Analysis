# 선형회귀분석 : LinearRegression

from sklearn.linear_model import LinearRegression
import statsmodels.api

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars[:3])

# hp(마력수)가 mpg(연비)에 영향을 미치는지, 인과관계가 있다면 연비에 미치는 영향값(추정치, 예측치)을 예측(정량적 분석)
x = mtcars[['hp']].values
y = mtcars[['mpg']].values
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.show()

fit_model = LinearRegression().fit(x, y)
print("slope :", fit_model.coef_[0])        # -0.06822828
print("intercept :",fit_model.intercept_)   # 30.09886054
# newy = -0.06822828 * newX + 30.09886054

pred = fit_model.predict(x)
print('예측값 : ', pred[:3].flatten())
print('예측값 : ', y[:3].flatten())

print()
# 모델성능 파악 시  R2 또는 RMSE
from sklearn.metrics import mean_squared_error
import numpy as np
lin_mse = mean_squared_error(y, pred)
lin_rmse = np.sqrt(lin_mse)
print('RMSE :', lin_rmse)

print('-----마력에 따른 연비 추정치-----')
new_hp = [[200]]
new_pred = fit_model.predict(new_hp)
print("%s 마력인 경우 연비 추정치는 %s"%(new_hp[0][0], new_pred[0][0]))














