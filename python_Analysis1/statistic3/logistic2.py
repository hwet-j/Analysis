# 로지스틱 회귀분석 : 날씨 예보 - 비가 올지 안올지 예보
import pandas as pd
from sklearn.model_selection._split import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("../testdata/weather.csv")
print(data.head(2), data.shape, data.columns)
data2 = pd.DataFrame()  # 빈 데이터 프레임 생성
data2 = data.drop(['Date','RainToday'], axis = 1)   # data에서 Date, RainToday를 제외함
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})    # Yes를 1로, No를 0으로
print(data2.head(5))

print()
# train(모델 학습) / test(모델성을 검정) 로 분리 : 과적합 방지
train, test = train_test_split(data2, test_size=0.3, random_state = 42)
print(train.shape, test.shape)

# 분류 모델
# myformula = 'RainTomorrow ~ MinTemp + MaxTemp + ........'    # 칼럼이 많으면 작성이 힘듬
col_sel = "+".join(train.columns.difference(['RainTomorrow']))    # RainTomorrow를 제외하고 train의 칼럼들을 +로 하여 나열 
myformula = 'RainTomorrow ~ ' + col_sel
print(myformula)
# model = smf.glm(formula = myformula, data = train, family = sm.families.Binomial()).fit()
model = smf.logit(formula = myformula, data = train).fit()
print(model)
# print(model.params)
print("예측값 : ", np.around(model.predict(test)[:5]))
print("실제값 : ", test['RainTomorrow'][:5])

# 정확도
con_mat = model.pred_table()    #  smf.logit()에서 지원. smf.glm()은 지원하지 않음
print('con_mat : \n', con_mat)  # 1행 1열은 1은 1로 , 1행 2열은 1을 0으로 예측 (즉, 대각선이 제대로 예측한 숫자)
print('분류 정확도 : ', (con_mat[0][0] + con_mat[1][1]) / len(train))    # 0.87109375
from sklearn.metrics import accuracy_score
pred = model.predict(test)  # sigmoid function에 의해 출력 
print('분류 정확도 :', accuracy_score(test['RainTomorrow'], np.around(pred)))    # 0.8727272727272727





