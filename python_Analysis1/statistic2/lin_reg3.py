# 선형회귀 모델 생성 후 처리 방법4 : linregress 사용. model O
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_iq = pd.read_csv("../testdata/score_iq.csv")
print(score_iq.head(3))
print(score_iq.info())

# iq(독립변수, x)가 score(종속변수, y)에 영향을 주는지 검정
# iq로 score(시험점수) 값 예측 - 정량적 분석

x = score_iq.iq
y = score_iq.score

# 상관계수 
print(np.corrcoef(x,y)) # numpy
print(score_iq.corr())  # pandas
    
# 두 변수는 인과관계가 있다고 보고, 선형회귀분석 진행
model = stats.linregress(x, y)
print(model)    
print('p값 :', model.pvalue) # 2.8476895206666614e-50 < 0.05 현재 모델은 유의하다.
print('기울기 :', model.slope) # 0.6514309527270089
print('절편 :', model.intercept)  # -2.8564471221976504
# y = 0.6514309527270089 * x + (-2.8564471221976504)
print('예측결과 : ', 0.6514309527270089 * 140 + (-2.8564471221976504))
print('예측결과 : ', 0.6514309527270089 * 125 + (-2.8564471221976504))
print('예측결과 : ', 0.6514309527270089 * 80 + (-2.8564471221976504))
print('예측결과 : ', 0.6514309527270089 * 155 + (-2.8564471221976504))
print('예측결과 : ', model.slope * 155 + (model.intercept))

# linregress는 predict()가 지원되지 않음. numpy의 ployval 이용
#print('예측결과 : ', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'])))
newdf = pd.DataFrame({'iq':[55,66,77,88,155]})
print('예측결과 : ', np.polyval([model.slope, model.intercept], newdf))




