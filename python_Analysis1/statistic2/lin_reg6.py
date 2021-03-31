# 선형회귀분석 : mtcars dataset. ols(). 모델 작성 후 추정치 얻기
import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
# print(mtcars.describe())
print(np.corrcoef(mtcars.hp, mtcars.mpg))
print(np.corrcoef(mtcars.wt, mtcars.mpg))
# print(mtcars.corr())

# 시각화
'''
plt.scatter(mtcars.hp, mtcars.mpg)
plt.xlabel('마력수')
plt.ylabel('연비')
slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)
plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
plt.show()
'''

print('\n단순 선형회귀------------')
result = smf.ols('mpg ~ hp', data = mtcars).fit()
#print(result.summary())
print(result.conf_int(alpha=0.05))
print(result.summary().tables[0])
print('마력수 110에 대한 연비 예측 : ', -0.0682 * 110 + 30.0989)  # 22.5969

print('\n다중 선형회귀------------')
result2 = smf.ols('mpg ~ hp + wt', data = mtcars).fit()
print(result2.summary())
# print(result2.conf_int(alpha=0.05))
# print(result2.summary().tables[0])
print('마력수 110 + 무게5 에 대한 연비 예측 : ', (-0.0318 * 110) + (-3.8778 * 5) + 37.2273)  # 14.3403

print('------추정치 구하기  차채 무게를 입력해 연비를 추정-------')
result3 = smf.ols('mpg ~ wt', data = mtcars).fit()
print(result3.summary())
print('결정계수 : ', result3.rsquared)  # 0.7528327 설명력이 우수한 모델이라 판단
pred = result3.predict()
print(pred) # 모든 자동차 차체무게에 대한 연비 추정치 출력

# 1개의 자료로 실제값과 예측값(추정값) 비교
print(mtcars.mpg[0])
print(pred[0])

data = {
    'mpg':mtcars.mpg,
    'mpg_pred':pred,
}
df = pd.DataFrame(data)
print(df)

print()
# 새로운 차체무게로 연비 추정하기
mtcars.wt = int(input('차체 무게입력 :'))
new_pred = result3.predict(pd.DataFrame(mtcars.wt))
print('차체무게:{}일때 예상연비는 {}'.format(mtcars.wt[0], new_pred[0]))

print()
# 차체 무게
new_wt = pd.DataFrame({'wt':[6, 3, 0.5]})
new_pred2 = result3.predict(new_wt)
print('예상연비 : \n', new_pred2)
print('예상연비 : \n', np.round(new_pred2.values, 2))








