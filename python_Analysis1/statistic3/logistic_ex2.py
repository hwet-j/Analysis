# [분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값1(착용X), 값2(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('../testdata/bodycheck.csv')
print(data.head(3), data.shape)

# x = data.loc[data['게임'] | data['TV시청']] 
x = data[['게임','TV시청']] 
y = data.안경유무
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (14, 6) (6, 6) (14,) (6,)

# 분류 모델 생성
model = LogisticRegression(C=1.0, random_state = 0) # C속성 : 모델에 패널티를 적용 (L2정규화) - 과적합 방지
model.fit(x_train, y_train) # 학습


# 분류예측 
y_pred = model.predict(x_test)  # 검정자료는 test
print('예측값 : ', y_pred)
print('실제값 : ', y_test)

# 분류 정확도 
print('총 개수 : %d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print()
print('분류 정확도 출력 1: %.3f'%accuracy_score(y_test, y_pred))

# confusion_matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred, labels = [0, 1]))    # [1, 0] 도 상관없음

# 입력 받은 새로운 값으로 예측하기
tmp1 = int(input("게임 시간 입력 :"))
tmp2 = int(input("Tv 시청 시간 입력 :"))

# predict() 함수로 결과 예측
predData = pd.DataFrame({'게임':[tmp1],'TV':[tmp2]})
y_pred2 = model.predict(predData)
print('y_pred2 : ', y_pred2)
result = ''

if y_pred2 == 0:
    result = "안경착용 X"
elif y_pred2 == 1:
    result = "안경착용 O"

print("게임시간이  %d시간이고  TV시청시간이 %d시간인 사람의 안경착용여부는: %s" %(tmp1,tmp2,result))   





