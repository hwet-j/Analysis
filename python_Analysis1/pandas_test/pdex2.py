# 재색인, NaN, bool 처리, 슬라이싱 관련 메소드, 연산....
from pandas import Series, DataFrame
import numpy as np

# Series의 재색인
data = Series([1,3,2], index=(1,4,2))
print(data)

data2 = data.reindex((1,2,4))
print(data2)

print('재색인할 떄 값 채워 넣기')
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)    # 대응 값이 없는 인덱스는 NaN(결측값)이 됨

print()
data3 = data2.reindex([0,1,2,3,4,5], fill_value = 333)  # 대응값이 없는 인덱스를 특정값으로 채움
print(data3)

print()
data3 = data2.reindex([0,1,2,3,4,5], method = 'ffill')  # 대응값이 없는 인덱스를 이전값으로 채움
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method = 'pad')  # 대응값이 없는 인덱스를 이전값으로 채움
print(data3)

print()
data3 = data2.reindex([0,1,2,3,4,5], method = 'bfill')  # 대응값이 없는 인덱스를 다음값으로 채움
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method = 'backfill')  # 대응값이 없는 인덱스를 다음값으로 채움
print(data3)

print('\nbool 처리')
df = DataFrame(np.arange(12).reshape(4,3), index = ['1월','2월','3월','4월'], columns = ['강남','강북','서초'])
print(df)
print(df['강남'])
print(df['강남'] > 3) # True, False 반환
print(df[df['강남'] > 3])
df[df < 3] = 0
print(df)

print() # 복수 인덱싱 loc: 라벨 지원, iloc: 숫자 지원
print(df.loc['3월', :])
print(df.loc['3월', ])
print()
print(df.loc[: '2월'])
print(df.loc[: '2월', ['서초']])

print()
print(df.iloc[2])
print(df.iloc[2, :])

print(df.iloc[:3])
print(df.iloc[:3, 2])
print(df.iloc[:3, 1:3])

print('\n\n ----연산 ----')
s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([4,5,6,7], index=['a','b','d','c'])
print(s1)
print(s2)

print(s1 + s2)  # 인덱스가 같은 경우만 연산. 불일치시 NaN,  ( -, *, / )
print('---------')
print(s1.add(s2))

print()
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울','대전','부산'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울','대전','제주','수원'])
print(df1)
print(df2)
print()
print(df1 + df2)
print(df1.add(df2))
print(df1.add(df2, fill_value = 0)) # NaN은 0으로 채운후 연산에 참여

print()
seri = df1.iloc[0]
print(seri)
print(df1)
print(df1 - seri)

print('\n함수--------')
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.NaN, None], [0.5, -1]])
print(df)
print(df.isnull())
print(df.notnull())

print(df.drop(1))  # 행 삭제
print(df.dropna())
print(df.dropna(how='all')) # 행의 모든 값이 NaN이면 삭제
print(df.dropna(how='any')) # 행의 값중 하나만 NaN이여도 삭제
print(df.dropna(axis='rows')) # NaN값이 있는 행 삭제  
print(df.dropna(axis='columns'))    # NaN값이 있는 컬럼 삭제
print(df.fillna(3)) # NaN값을 지정해준값으로 채워줌
# print(df.dropna(subset = ['one']))

print()
print(df.sum()) # 열단위 합
print(df.sum(axis=0))
print()
print(df.sum(axis = 1)) # 행단위

print(df.mean(axis=1))
print(df.mean(axis=1,skipna=False))
print(df.mean(axis=1,skipna=True))
print(df.mean(axis=0,skipna=True))
print()
print(df.describe())    # 요약 통계량
print(df.info())    # 구조 확인

print()
words = Series(['봄', '여름', '가을', '봄'])
print(words.describe())
#print(words.info()) # AttributeError: 'Series' object has no attribute 'info'








