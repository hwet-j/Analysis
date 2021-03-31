# 나이베이즈 분류 모델 : feature가 주어졌을 때 label의 확률을 구함. P(L|Feature)
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]
print(x)
y = np.array([1,3,5,7,9])
print(y)

model = GaussianNB().fit(x, y)
pred = model.predict(x)
print(pred)
print('acc : ', metrics.accuracy_score(y, pred))

# new data
new_x = np.array([[0.5],[2.3],[12],[0.1]])
new_pred = model.predict(new_x)
print('new_pred : ', new_pred)

print('feature 데이터를 One-hot 인코딩 -----------')
x = '1,2,3,4,5'
x = x.split(',')
x = np.eye(len(x))
print(x)
y = np.array([1,3,5,7,9])
model = GaussianNB().fit(x,y)
pred = model.predict(x)
print(pred)
print('acc : ', metrics.accuracy_score(y, pred))

print()
from sklearn.preprocessing import OneHotEncoder
x = '1,2,3,4,5'
x = x.split(',')
x = np.array(x)
x = x[:, np.newaxis]
one_hot = OneHotEncoder(categories = 'auto')
x = one_hot.fit_transform(x).toarray()
print(x)
y = np.array([1,3,5,7,9])

model = GaussianNB().fit(x,y)
pred = model.predict(x)
print(pred)
print('acc : ', metrics.accuracy_score(y, pred))




