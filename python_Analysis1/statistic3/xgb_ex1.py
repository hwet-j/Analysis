# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7가지의 label(Type)로 분리된다.
# RI    Na    Mg    Al    Si    K    Ca    Ba    Fe   Type
# glass.csv 파일을 읽어 분류 작업을 수행하시오.
import pandas as pd
from sklearn import metrics, model_selection
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance


data = pd.read_csv('glass.csv')
# print(data.head(3), data.shape) # (214, 10)
print(data.corr())
df_x = data.drop(['Type'], axis=1)
# print(df_x.head(3))
df_y = data['Type']
# print(df_y.head(3))

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3,random_state=123)
print()
'''
# 스케일링
sc = StandardScaler()
sc.fit(x_train, x_test) 
x_train = sc.transform(x_train)  
x_test = sc.transform(x_test)
'''

# XGBClassifier 모델
import xgboost as xgb
model = xgb.XGBClassifier(booster='gbtree', max_depth=4, n_estimators=100)
model.fit(x_train, y_train)

# 예측
y_pred = model.predict(x_test)
# print('예측값 : ', y_pred[:5])
# print('실제값 : ', np.array(y_test[:5]))
print('정확도 : ', metrics.accuracy_score(y_test, y_pred)) # 0.73846153

# 분류 정확도
print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print()
print('분류 정확도 출력 : %.3f'%accuracy_score(y_test, y_pred)) # 0.738
print('분류 정확도 출력 test:', model.score(x_test, y_test))  # test    0.7384615
print('분류 정확도 출력 train:', model.score(x_train, y_train)) # train    1.0


# 중요도 체크
print("RandomForest Modeling")
rf_model = RandomForestClassifier(n_estimators=10,criterion="entropy",n_jobs=1,random_state=123)
rf_model.fit(x_train,y_train)
# 데이터 셋 분할 (by cross validation) 및 분류 정확도 평균 계산
# cv = model_selection.cross_val_score(rf_model,df_x, df_y, cv=6)   
# print("cross validation 정확도 평균:",cv.mean())
col = pd.DataFrame(x_train,columns=df_x.columns)
print(col.columns)

varDic = {'var':col.columns,'imp': rf_model.feature_importances_}
imp = pd.DataFrame(varDic)
imp = imp.sort_values(by='imp', ascending=False)[0:9]
print(imp)  

# print("RandomForest Importance barChart")
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
plt.rc('font', family="malgun gothic")
plt.title("특성 중요도")
plt.bar(range(x_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
x_train = pd.DataFrame(x_train)
plt.xticks(np.arange(x_train.shape[1]), tuple(imp["var"]))
plt.xlim([-1, x_train.shape[1]])
plt.show()

# print("xgboost Importance barChart")
xgb.plot_importance(model)
plt.show()

# 정확도 (rf)
pred = rf_model.predict(x_test)
acc = metrics.accuracy_score(y_test,pred)
print("정확도:", acc)


