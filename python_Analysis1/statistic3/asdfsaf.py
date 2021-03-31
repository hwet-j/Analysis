'''
[XGBoost 문제] 
kaggle.com이 제공하는 'glass datasets'
유리 식별 데이터베이스로 여러 가지 특징들에 의해 7가지의 label(Type)로 분리된다.
RI    Na    Mg    Al    Si    K    Ca    Ba    Fe    
 Type
                          ...
glass.csv 파일을 읽어 분류 작업을 수행하시오.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import xgboost as xgb  
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv('../testdata/glass.csv')
print(data.shape) # (214, 10)
print('타입 :', data['Type'].head(5))
print('데이터 열 이름 :', data.columns)

x = data[data.columns.difference(['Type'])]
y = data['Type']

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=8)

# scaling
sc = StandardScaler()
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test) 

#model = xgb.XGBClassifier(booster='gbtree', max_depth=6, n_estimators=100)
model = RandomForestClassifier(n_estimators=100) 
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:10])
print('실제값 : ', np.array(y_test[:10]))

print('정확도 : ', metrics.accuracy_score(y_test, y_pred))

# 중요 변수 알아보기
print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))

def plot_feature_importances(model):   # 특성 중요도 시각화
    n_features = x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(model)


