# [Randomforest 문제] 
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd 
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/patient.csv")
print(data.head(5))
print(data.info())
y = data["STA"]
x = data[["AGE" ,'SEX' , 'RACE' , 'SER' , 'CAN' , 'CRN' ,'INF' , 'CPR' , 'HRA']]   
x_age = pd.get_dummies(data['SEX'], prefix = "성별") 
x_race = pd.get_dummies(data['RACE'], prefix = "인종") 
x_ser = pd.get_dummies(data['SER'], prefix = "치료") 
x_can = pd.get_dummies(data['CAN'], prefix = "암유무") 
x_crn = pd.get_dummies(data['CRN'], prefix = "CRN") 
x_inf = pd.get_dummies(data['INF'], prefix = "감염여부") 
x_cpr = pd.get_dummies(data['CPR'], prefix = "CPR여부") 
x = pd.concat([x["AGE"],x["HRA"],x_age,x_race,x_ser,x_can,x_crn,x_inf,x_cpr],axis=1)
print(x[:3])
print(x.columns)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)
print()
#스케일링
sc = StandardScaler()
sc.fit(x_train, x_test) 
x_train = sc.transform(x_train)  
x_test = sc.transform(x_test)  

print("Modeling")
rf_model = RandomForestClassifier(n_estimators=10,criterion="entropy",n_jobs=1,random_state=123)
rf_model.fit(x_train,y_train)
# 데이터 셋 분할 (by cross validation) 및 분류 정확도 평균 계산
cv = model_selection.cross_val_score(rf_model,x,y,cv=6)   
print("cross validation 정확도 평균:",cv.mean())
col = pd.DataFrame(x_train,columns=x.columns)
print(col.columns)

varDic = {'var':col.columns,'imp': rf_model.feature_importances_}
imp = pd.DataFrame(varDic)
imp = imp.sort_values(by='imp', ascending=False)[0:17]
print(imp)  

print("Importance barChart")
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

# 예측 정확도
pred = rf_model.predict(x_test)
acc = metrics.accuracy_score(y_test,pred)
print("정확도:", acc)

