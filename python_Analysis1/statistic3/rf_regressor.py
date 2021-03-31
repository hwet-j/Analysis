# RandomForest : 정량적인 분석 모델 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston    # 보스턴 집값
from sklearn.metrics import r2_score    # 결정계수


boston = load_boston()
print(boston.DESCR)
# data와 target이 따로 (boston자료는 dataframe형태가아닌 array형태로 와서) 존재하기 때문에 dataframe형식으로 만들어준다.
dfx = pd.DataFrame(boston.data, columns = boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=['MEDV']) # 집의 평균값 
print(dfx.head(3), dfx.shape)   # (506, 13)
print(dfy.head(3), dfy.shape)   # (506, 1)

df = pd.concat([dfx, dfy], axis=1)  # 동일한 형태의 dataframe 합치기
print(df.head(3), df.shape)

# pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)
print(df.corr())

# 시각화 
# cols = ['MEDV','RM','AGE','LSTAT']
# sns.pairplot(df[cols])
# plt.show()


x = df[['LSTAT']].values    # 영향이 가장 큰 값
y = df['MEDV'].values
print(x[:2])
print(y[:2])

# 실습1
model = DecisionTreeRegressor(max_depth=3).fit(x, y)
print('predict : ', model.predict(x)[:5])
print('real : ', y[:5])
r2 = r2_score(y, model.predict(x))
print('결정계수(R2, 설명력) :', r2)    # 0.69938330

# 실습2
model2 = RandomForestRegressor(n_estimators = 1000, criterion = 'mse').fit(x, y)    # mse 평균제곱오차
print('predict : ', model2.predict(x)[:5])
print('real : ', y[:5])
r2_1 = r2_score(y, model2.predict(x))
print('결정계수(R2, 설명력)2 :', r2_1) # 0.90984572

print('\n학습 검정 자료로 분리 ')
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, random_state = 123)
model2.fit(train_x, train_y)

# train과 test설명력의 차이가 얼마나지 않아야함
r2_train = r2_score(train_y, model2.predict(train_x))   # train에 대한 설명력
print('train에 대한 설명력 :', r2_train)  # 0.908974

r2_test = r2_score(test_y, model2.predict(test_x))   # test에 대한 설명력
print('test에 대한 설명력 :', r2_test)    # 0.57978    독립변수의 수를 늘려주면 결과는 달라짐 

# 시각화
from matplotlib import style
style.use('seaborn-talk')
plt.scatter(x, y, c='lightgray', label = 'train data')
plt.scatter(test_x, model2.predict(test_x), c='r', label='predict data, $R^2=%.2f$'%r2_test)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# 새값으로 예측
import numpy as np
print(test_x[:3])
x_new = [[50.11],[ 26.53],[ 1.76]]
print('예상 집값 : ', model2.predict(x_new))



