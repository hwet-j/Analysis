# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭하기 
# 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청시간과 운동량 대한 데이터는 아래와 같다.
#  - 지상파 시청시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  - 지상파 시청시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#     참고로 결측치는 해당 칼럼의 평균값을 사용하기로 한다. 운동 칼럼에 대해 이상치가 있는 행은 제거.
import scipy.stats as stats
import pandas as pd
import numpy as np


data = pd.read_csv('lin_reg_ex.txt',delimiter=',')
data = data.set_index('구분')
print(data.head(3))
data['지상파'] = data['지상파'].fillna(data['지상파'].mean()) # 결측치 평균으로 대체

# 이상치 제거작업(특정행에서)
data_1 = data['운동'].quantile(0.25)
data_3 = data['운동'].quantile(0.75)
IQR = data_3 - data_1
search_df = data[(data['운동'] < (data_1 - 1.5 * IQR)) | (data['운동'] > (data_3 + 1.5 * IQR))]
# print(search_df)    # 이상치 존재행 출력
data = data.drop(search_df.index, axis=0)   # 이상치 존재행 삭제
# print(data)    # 확인

x = data.지상파
y = data.운동
z = data.종편
model = stats.linregress(x, y)
model2 = stats.linregress(x, z)
try:
    number = float(input('1.지상파 시청시간을 입력하세요(운동):'))
    number2 = float(input('2.지상파 시청시간을 입력하세요 (종편):'))
except:
    print("숫자가 입력되지않았습니다.")
# print('p값 :', model.pvalue) # 0.2888491285513351
# print('기울기 :', model.slope) # 1.6677694999166923
# print('절편 :', model.intercept)  #  0.6184430442731452
print('예측결과(운동) : ', model.slope * number + (model.intercept))
print('예측결과2(종편) : ', model2.slope * number + (model2.intercept))

