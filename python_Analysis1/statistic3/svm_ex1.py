# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv
# 
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
# 
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)

import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Heart.csv')
data['Ca'] = data['Ca'].fillna(data['Ca'].mean())   # Na값 (Ca에만 존재해서 여기만 적용) 평균값으로 
data = data.drop(['ChestPain','Thal'], axis=1)  # 컬럼 ChestPain,  Thal 제외
print(data.head(3), data.shape)
df = data.drop(['AHD'], axis=1)
# 칼럼을 정규화
AHD = data['AHD']
print(AHD)

print(df.head(5), df.shape) # (303, 12)
print()
AHD = AHD.map({'Yes':0,'No':1})
print(AHD[:5], AHD.shape)   # (303,)

# train / test
data_train, data_test, label_train, label_test = train_test_split(df, AHD)
print(data_train.shape, data_test.shape)    # (227, 12) (76, 12)

# model 
model = svm.SVC(C=0.01).fit(data_train, label_train)
print(model)

# 학습한 데이터의 결과가 신뢰성이 있는지 확인하기 위해 교차검증 ----------
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, df, AHD, cv = 6)
print('각각의 검증 결과', cross_vali)
print('평균 검증 결과', cross_vali.mean())
# --------------------------------------------------

pred = model.predict(data_test)
ac_score = metrics.accuracy_score(label_test, pred)
print('분류 정확도 : ',ac_score) # 분류 정확도 :  0.5789473
print(metrics.classification_report(label_test, pred))



