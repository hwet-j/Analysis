# [Randomforest 문제1] 
# 중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
# 종속변수 STA에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 
# 예제 파일 : https://github.com/pykwon  ==>  patient.csv
# <변수설명>
#   STA : 환자 생사 여부
#   AGE : 나이
#   SEX : 성별
#   RACE : 인종
#   SER : 중환자 치료실에서 받은 치료
#   CAN : 암 존재 여부
#   INF : 중환자 치료실에서의 감염 여부
#   CPR : 중환자 치료실 도착 전 CPR여부
#   HRA : 중환자 치료실에서의 심박수

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/patient.csv")
print(df.head(3), df.shape) # (200, 11)
print(df.columns)

df_x = df[['ID', 'AGE', 'SEX', 'RACE', 'SER', 'CAN', 'CRN', 'INF', 'CPR', 'HRA']]   # 사용할 데이터만 뽑아옴
print(df_x.head(3), df_x.shape) # (200, 10)
df_y = df['STA']
print(df_y.head(3), df_y.shape) # (200,)



# train/ test
(train_x, test_x, train_y, test_y) = train_test_split(df_x, df_y)

# model 
model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
fit_model = model.fit(train_x, train_y)

pred = fit_model.predict(test_x)
print('예측값  :', pred[:10])
print('실제값  :', test_y[:10].ravel())

# 정확도 
print('acc : ', sum(test_y == pred) / len(test_y))
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(test_y, pred))

'''
# 참고 : 중요 변수 알아보기
print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
def plot_feature_importances(model):   # 특성 중요도 시각화
    n_features = df_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df_x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(model)
# HRA, CPR, INF, CAN, AGE, ID
'''

# 주요변수라 판단되는 것들만 데이터로
print('주요변수--------------------------------')
df_x2 = df[['ID', 'AGE', 'INF', 'CPR', 'HRA', 'CAN']]   # 주요변수라 판단되는 데이터만 뽑아옴
print(df_x2.head(3), df_x2.shape) # (200, 6)
df_y2 = df['STA']
print(df_y2.head(3), df_y2.shape) # (200,)

# train/ test
(train_x2, test_x2, train_y2, test_y2) = train_test_split(df_x2, df_y2)

# model 
model2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
fit_model2 = model2.fit(train_x2, train_y2)

pred2 = fit_model2.predict(test_x2)
print('예측값  :', pred2[:10])
print('실제값  :', test_y2[:10].ravel())

# 정확도 
print('acc : ', sum(test_y2 == pred2) / len(test_y2))
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(test_y2, pred2))


