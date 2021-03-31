# pandas 문제 3)  타이타닉 승객 데이터를 사용하여 아래의 물음에 답하시오.
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv')
#print(df)

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
# 1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
result_cut = pd.cut(df['Age'], bins, labels = labels)
print(result_cut.value_counts())

# 2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다.
# 2-1)
print(df.pivot_table(values='Survived', index='Sex', columns='Pclass', fill_value=0))
# 2-2)
print('\n\n')
df_n = df.pivot_table(values='Survived', index=['Sex', 'Age'], columns='Pclass', fill_value=0)
print(round(df_n * 100, 2))






