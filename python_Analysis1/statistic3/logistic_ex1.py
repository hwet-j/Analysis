import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import accuracy_score

# 문제1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라
data = pd.read_csv("logistic_ex1.txt")
# print(data.head(2), data.shape, data.columns)
data = data.query("요일 == '토' | 요일 == '일'")  #  요일이 토,일 인 값만 
# data = data.loc[(data['요일']=='토') | (data['요일']=='일')]     #  요일이 토,일 인 값만 
# print(data, data.shape)

# 분류 모델
# model = smf.glm(formula = myformula, data = train, family = sm.families.Binomial()).fit()
model = smf.logit(formula = '외식유무 ~ 소득수준', data = data).fit()
pred = model.predict(data)
print("예측값 : ", np.around(pred))
print("실제값 : ", data['외식유무'])
print('분류 정확도 : ', accuracy_score(data['외식유무'], np.around(pred)))  # 0.9047619047619048

print()
new_input_data = pd.DataFrame({'소득수준':[int(input('소득수준  입력: '))]})
print('외식 유무 :', np.rint(model.predict(new_input_data)))
print('외식을 함' if np.rint(model.predict(new_input_data))[0] == 1 else '외식안함')





