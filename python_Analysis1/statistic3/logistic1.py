# 로지스틱 회귀분석 : 이항 분류 분석 :  logit(), glm()
# 독립변수: 연속형, 종속변수:범주형
# 출력된 연속 자료에 대해 odds ratio -> logit -> sigmoid function으로 이항분류

import math
import numpy as np

def sigFunc(x):
    return 1 / (1 + math.exp(-x))

print(sigFunc(0.6))
print(sigFunc(0.2))
print(sigFunc(6))
print(sigFunc(-6))
print(np.around(sigFunc(6)))
print(np.around(sigFunc(-6)))

print('-------------------')
import statsmodels.formula.api as smf
import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars['am'].unique())

print('방법1 : logit() -----------------')
formula = 'am ~ mpg + hp'
result = smf.logit(formula = formula, data = mtcars).fit()
print(result.summary())


# print('방법2 : glm() ------------------')
# print('예측값 : ', result.predict())
pred = result.predict(mtcars[:10])
print('예측값 : ', np.around(pred))
print('실제값 : ', mtcars['am'][:10])

print()
conf_tab = result.pred_table()
print(conf_tab)
print('분류 정확도  : ', (16 + 10) / len(mtcars))
print('분류 정확도  : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcars))
from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcars)
print('분류 정확도 : ', accuracy_score(mtcars['am'], np.around(pred2)))

print('\n방법1 : glm() ')
result2 = smf.glm(formula = formula, data = mtcars, family = sm.families.Binomial()).fit()
print(result2)
# print(result2.summary())
glm_pred = result2.predict(mtcars[:5])
# print("glm 예측값 : ", glm_pred)
print("glm 예측값 : ", np.around(glm_pred))
print('실제값 : ', mtcars['am'][:5])
glm_pred2 = result2.predict(mtcars) # 분류정확도를 5개만으로 확인하지 않고 전체데이터로 확인하기 위함.
print('분류 정확도 : ', accuracy_score(mtcars['am'], np.around(glm_pred2)))

print('새로운 값을 분류')
newdf = mtcars.iloc[:2].copy()
newdf['mpg'] =  [10, 30]
newdf['hp'] =  [100, 130]
print(newdf)


glm_pred_new = result2.predict(newdf)
print('새로운 값 분류 결과 : ', np.around(glm_pred_new))
print('새로운 값 분류 결과 : ', np.rint(glm_pred_new))

print()
import pandas as pd
newdf2 = pd.DataFrame({'mpg':[10,35], 'hp':[100,145]})
glm_pred_new2 = result2.predict(newdf2)
print('새로운 값 분류 결과 : ', np.around(glm_pred_new2))



