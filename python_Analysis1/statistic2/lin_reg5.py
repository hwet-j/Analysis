# 선형회귀분석 : iris dataset. ols() 상관관계 약한 변수로 모델 / 상관관계가 강한 변수로 모델
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(3))
print(iris.corr())

# 단순 선형회귀 모델 : r:-0.117570 (sepal_length,  sepal_width)
result = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
# print(result.summary())
print('R-squared :', result.rsquared)   # 0.0138226 의미없는 모델
print('p-value :', result.pvalues)  # 1.518983e-01


print('-----------')
# 단순 선형회귀 모델 : r: 0.871754 (sepal_length,  petal_length)
result2 = smf.ols(formula = 'sepal_length ~ petal_length', data = iris).fit()
# print(result2.summary())
print('R-squared :', result2.rsquared)   # 0.759 의미있는 모델
print('p-value :', result2.pvalues)     # 1.038667e-47 < 0.05
print()
pred = result2.predict()
print('실제값 : ', iris.sepal_length[0])   # 5.1
print('예측값 : ', pred[0])    # 4.879

# 새로운 데이터로 예측
print(iris.petal_length[1:5])
new_data = pd.DataFrame({'petal_length':[1.4, 0.5, 8.5, 12.123]})
y_pred_new = result2.predict(new_data)
print('새로운 데이터로 sepal_length 예측 :\n', y_pred_new)

print('\n\n------다중 선형회귀--------')
result3 = smf.ols(formula = 'sepal_length ~ petal_length + petal_width', data = iris).fit()
# print(result3.summary())
print('R-squared :', result3.rsquared)   # 0.766261 의미있는 모델
print('p-value :', result3.pvalues)     # petal_length : 9.414477e-13 < 0.05, petal_width : 4.827246e-02 < 0.05
print()
# 새로운데이터로 예측2
new_data2 = pd.DataFrame({'petal_length':[8.5, 12.12], 'petal_width':[8.5, 12.5]})
y_pred_new2 = result3.predict(new_data2)
print('새로운 데이터로 sepal_length 예측: \n', y_pred_new2)


