# 선형회귀분석 : LinearRegression
# 과적합 방지를 위해 Ridge, Lasso, ElasticNet 

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
'''
print(iris.data)
print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target)
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
'''

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_names'] = iris.target_names[iris.target]
print(iris_df.head(3), ' ', iris_df.shape)      # (150, 6)

# train / test 분리
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris_df, test_size= 0.3)
print(train_set.head(3),' ', train_set.shape)   # (105, 6)
print(test_set.head(3), ' ', test_set.shape)    # (45, 6)

print('LinearRegression---------')
# 정규화 선형회귀방법은 선형회귀계수(weight)에 대한 제약조건을 추가함으로 해서, 모형이 과도하게 최적화(오버피팅)되는 현상을 방지할 수 있다.
from sklearn.linear_model import LinearRegression as lm
import matplotlib.pyplot as plt
# print(train_set.iloc[:, [2]])   #  petal length
# print(train_set.iloc[:, [3]])   #  petal width

model_ols = lm().fit(X = train_set.iloc[:, [2]], y = train_set.iloc[:, [3]])
print(model_ols.coef_[0])
print(model_ols.intercept_)
pred = model_ols.predict(test_set.iloc[:, [3]])
# print('ols_pred : ', model_ols.predict(test_set.iloc[:, [3]]))
print('ols_pred : ', pred[:5])
print('ols_real : ', test_set.iloc[:, [3]][:5])

print()

print('\nRidge')
# 회귀분석 방법 - Ridge: alpha값을 조정(가중치 제곱합을 최소화)하여 과대/과소적합을 피한다. 다중공선성 문제 처리에 효과적.
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10).fit(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])
 
#점수
print(model_ridge.score(X=train_set.iloc[:, [2]], y=train_set.iloc[:, [3]])) #0.91923658601
print(model_ridge.score(X=test_set.iloc[:, [2]], y=test_set.iloc[:, [3]]))   #0.935219182367
print('ridge predict : ', model_ridge.predict(test_set.iloc[:, [2]]))
plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='red')
plt.plot(test_set.iloc[:, [2]], model_ridge.predict(test_set.iloc[:, [2]]))
plt.show()



print('\nLasso')
# 회귀분석 방법 - Lasso: alpha값을 조정(가중치 절대값의 합을 최소화)하여 과대/과소적합을 피한다.
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.1, max_iter=1000).fit(X=train_set.iloc[:, [0,1,2]], y=train_set.iloc[:, [3]])
 
#점수
print(model_lasso.score(X=train_set.iloc[:, [0,1,2]], y=train_set.iloc[:, [3]])) #0.921241848687
print(model_lasso.score(X=test_set.iloc[:, [0,1,2]], y=test_set.iloc[:, [3]]))   #0.913186971647
print('사용한 특성수 : ', np.sum(model_lasso.coef_ != 0))   # 사용한 특성수 :  1
plt.scatter(train_set.iloc[:, [2]], train_set.iloc[:, [3]],  color='red')
plt.plot(test_set.iloc[:, [2]], model_ridge.predict(test_set.iloc[:, [2]]))
plt.show()


# 회귀분석 방법 4 - Elastic Net 회귀모형 : Ridge + Lasso 가중치 제곱합을 최소화, 가중치 절대값의 합을 최소화 두 가지를 동시에 제약조건으로 사용
from sklearn.linear_model import ElasticNet





