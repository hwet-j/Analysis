# 기술 통계
'''
기술통계(descriptive statistics)란 수집한 데이터의 특성을 표현하고 요약하는 통계 기법이다. 
기술통계는 샘플(전체 자료일수도 있다)이 있으면, 그 자료들에 대해  수치적으로 요약정보를 표현하거나, 
데이터 시각화를 한다. 
즉, 자료의 특징을 파악하는 관점으로 보면 된다. 평균, 분산, 표준편차 등이 기술통계에 속한다.
'''
# 도수분포표
import pandas as pd

frame = pd.read_csv('../testdata/ex_studentlist.csv')
print(frame.head(2))
print(frame.info())
print('나이 : ', frame['age'].mean())
print('나이 : ', frame['age'].var())
print('나이 : ', frame['age'].std())
print('혈액형 : ', frame['bloodtype'].unique())
print(frame.describe())

print()
# 혈액형별 인원수 
data1 = frame.groupby(['bloodtype'])['bloodtype'].count()
print('혈액형별 인원수 : ', data1)

print()
data2 = pd.crosstab(index = frame['bloodtype'], columns='count')
print('혈액형별 인원수 : ', data2)

print()
# 성별, 혈액형별 인원수 
data3 = pd.crosstab(index = frame['bloodtype'], columns=frame['sex'], margins=True) # 소계
data3.columns = ['남','여','행합']
data3.index = ['A','AB','B','O','열합']
print('성별, 혈액형별 인원수  : \n', data3)
print()








