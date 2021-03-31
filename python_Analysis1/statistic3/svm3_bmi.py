# BMI의 계산방법을 이용하여 많은 양의 자료를 생성 한 후 분류 모델로 처리
# 신체질량지수(BMI) = 체중(kg) / [신장(m)]^2
# 과체중 : 23 - 24.9
# 정상 : 18.5 - 22.9
# 저체중 : 18.5 미만

import random
'''
def calc_bmi(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 23: return 'normal'
    return 'fat'

# print(calc_bmi(180, 74))
fp = open('bmi.csv', 'w') 
fp.write('height,weight,label\n')
cnt = {'thin':0,'normal':0,'fat':0}

for i in range(50000):
    h = random.randint(150, 200)
    w = random.randint(35, 100)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h,w,label))
    

fp.close()
print('good')
'''

# BMI dataset으로 분류
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

tbl = pd.read_csv('bmi.csv')
print(tbl.columns)
# 칼럼을 정규화
label = tbl['label']
print(label)

w = tbl['weight'] / 100
h = tbl['height'] / 200
wh = pd.concat([w, h], axis=1)
print(wh.head(5), wh.shape) # (50000, 2)
print()
label = label.map({'thin':0,'normal':1,'fat':2})
print(label[:5], label.shape)   # (50000,)

# train / test
data_train, data_test, label_train, label_test = train_test_split(wh, label)
print(data_train.shape, data_test.shape)    # (37500, 2) (12500, 2)

# model 
model = svm.SVC(C=0.01).fit(data_train, label_train)
# model = svm.LinearSVC().fit(data_train, label_train)
print(model)

# 학습한 데이터의 결과가 신뢰성이 있는지 확인하기 위해 교차검증 ----------
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, wh, label, cv = 3)
print('각각의 검증 결과', cross_vali)
print('평균 검증 결과', cross_vali.mean())
# --------------------------------------------------

pred = model.predict(data_test)
ac_score = metrics.accuracy_score(label_test, pred)
print('분류 정확도 : ',ac_score)
print(metrics.classification_report(label_test, pred))

tbl2 = pd.read_csv('bmi.csv', index_col = 2)
print(tbl2[:3])

# 시각화
def scatter_func(lbl, color):
    b = tbl2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c = color, label = lbl)


fig = plt.figure()  # savefig(저장)하기위해 써준다. (없어도되지만 있는게 좋음)
scatter_func('fat', 'red')
scatter_func('normal', 'yellow')
scatter_func('thin', 'blue')
plt.legend()
plt.savefig('bmi_test.png')
plt.show()









