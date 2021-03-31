# 회귀분석 문제 2) 
# github.com/pykwon/python에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측
import scipy.stats as stats
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

data = pd.read_csv('../testdata/student.csv')
# print(data)

# 피어슨 상관계수
# print(data.corr())    # 국어/수학의 상관관계 0.766263

# - 국어 점수를 입력하면 수학 점수 예측
result = smf.ols('수학 ~ 국어', data = data).fit()
# print(result.summary())
# print(result.params.Intercept)
# print(result.params.국어)

# 32.1069 절편
# 0.5705 기울기
score = int(input('국어 점수입력 :'))
# print('예측값 :', 0.5705 * score + 32.1069)
new_df = pd.DataFrame({'국어':[score]}) 
# print(new_df)
new_pred = result.predict(new_df)
print('예측값 : \n', new_pred)

#   - 국어, 영어 점수를 입력하면 수학 점수 예측
result2 = smf.ols('수학 ~ 국어  + 영어', data = data).fit()
print(result2.summary())
# Adj. R-squared: 0.619 (다중회귀분석에서는 이값을 보며, 설명력이 약 62%라는 뜻)
# 국어 : 0.663 > 0.05, 영어 : 0.074 > 0.05 사실상 유의미한 데이터가 아님.
# 국어 0.1158, 영어 0.5942 , 절편 22.6238
score2 = int(input('영어 점수입력 :'))

print('예측값 :', 0.1158 * score + 0.5942 * score2 + 22.6238)

