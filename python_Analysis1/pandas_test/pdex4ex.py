import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv')
print(df.head(), '\n')

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
df.Age = pd.cut(df.Age, bins, labels = labels)
print('1) \n', df.Age.value_counts(), '\n')

df_n = df.pivot_table(index = ['Sex'], columns = 'Pclass', \
                        values = 'Survived', fill_value = 0)
print('2) \n', round(df_n * 100, 2))

df_n = df.pivot_table(index = ['Sex', 'Age'], columns = 'Pclass', \
                        values = 'Survived', fill_value = 0)
print('2) \n', round(df_n * 100, 2))
print('---------------------')

"""
pandas 문제 4)
  https://github.com/pykwon/python/tree/master/testdata_utf8
 
  1) human.csv 파일을 읽어 아래와 같이 처리하시오.
      - Group이 NA인 행은 삭제
      - Career, Score 칼럼을 추출하여 데이터프레임을 작성
      - Career, Score 칼럼의 평균계산
"""
human_df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/human.csv')
print(human_df.head(2))
print(human_df.info())
human_df = human_df.rename(columns=lambda x: x.strip())  # 칼럼명 앞에 공백 제거
print(human_df.head(2))
print(human_df.info())

print('---')
human_df['Group'] = human_df['Group'].str.strip()
human_df = human_df[human_df['Group']!='NA']
print(human_df.head(5),"\n")

#cs_df2 = human_df[human_df.columns[2:4]]
cs_df2 = human_df.loc[:, ['Career', 'Score']]
print(cs_df2.head(5),"\n")
print(cs_df2.mean())



# 2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
#      - 파일 정보 확인
#      - 앞에서 3개의 행만 출력
#      - 요약 통계량 보기
#      - 흡연자, 비흡연자 수를 계산  : value_count()
#      - 요일을 가진 칼럼의 유일한 값 출력  : unique()

tips_df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/tips.csv')
print(tips_df.info(),"\n")
print(tips_df.head(3),"\n")
print(tips_df.describe(),"\n")
print(pd.value_counts(tips_df['smoker']),"\n")
print(pd.unique(tips_df['day']))
