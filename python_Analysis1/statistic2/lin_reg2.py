# 기계학습 (지도학습) : 학습을 통해 모델 생성후, 새로운 데이터에 대한 예측 및 분류
# 회귀분석 : 각각의 데이터에 대한 잔차제급합이 최소가 되는 추세선을 만들고, 이를 통해 독립변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수 : 연속형, 종속변수 : 연속형. 두 변수는 상관관계가 있어야 하고 나아가서는 인과관계가 있어야 한다.
# 기본 충족 조건 : 선형성, 잔차정규성, 잔차독립성, 등분산성, 다중공선성 
# 정량적인 모델을 생성

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np

np.random.seed(12)
# 모델 생성 맛보기
# 방법1 : make_regression을 사용. model X
x, y, coef = make_regression(n_samples = 50, n_features = 1, bias=100, coef=True)
print(x)
print(y)
print(coef) # 89.47430739278907 기울기
# 회귀식 y = a + bx  y = 100 + 89.47430739278907 * x
y_pred = 100 + 89.47430739278907 * -1.70073563
print('y_pred : ', y_pred)

y_pred_new = 100 + 89.47430739278907 * 66   # 새로운 값에 대해 예측 결과
print('y_pred_new : ', y_pred_new)

xx = x
yy = y

print()
# 방법2 : LinearRegression을 사용. model O
from sklearn.linear_model import LinearRegression
model = LinearRegression()
fit_model = model.fit(xx, yy)   # 학습데이터로 모형 추정 : 절편, 기울기 얻음
print(fit_model.coef_)    # 기울기
print(fit_model.intercept_)    # 절편
# 예측값 확인 함수
print()
y_new = fit_model.predict(xx[[0]])
print('y_new : ', y_new)
y_new2 = fit_model.predict([[66]])
print('y_new2 : ', y_new2)

print()
# 방법 3 : ols 사용, model O
import statsmodels.formula.api as smf
import pandas as pd
x1 = xx.flatten()   # 차원 축소
print(x1.shape)
y1 = yy
print(y1.shape)

data = np.array([x1, y1])
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(3))

model2 = smf.ols(formula = 'y1 ~ x1', data = df).fit()
print(model2.summary())

# 예측값 확인 함수
print(x1[:2])   # [-1.70073563 -0.67794537]
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]}) # 기존 자료로 검정
print(new_df)
new_pred = model2.predict(new_df)
print('new_pred : \n', new_pred)

# 전혀 새로운 값에 대한 예측 결과 확인
new2_df = pd.DataFrame({'x1':[123, -2.34567]})
new2_pred = model2.predict(new2_df)
print('new2_pred : \n', new2_pred)








