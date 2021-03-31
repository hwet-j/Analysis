# 교차 테이블 (교차표)
import pandas as pd

y_true = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2])
y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2])

result = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(result)

#------------------------------------------------------
print()
print()
# 인구통계 dataset 읽기
des = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/descriptive.csv') 
print(des.info())

# 5개 칼럼만 선택하여 data frame 생성 
data = des[['resident','gender','age','level','pass']]
print(data[:5])

# 지역과 성별 칼럼 교차테이블 
table = pd.crosstab(data.resident, data.gender)
print(table)

# 지역과 성별 칼럼 기준 - 학력수준 교차테이블 
table = pd.crosstab([data.resident, data.gender], data.level)
print(table)




