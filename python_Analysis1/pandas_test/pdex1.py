# pandas : 고수준의 자료구조(Series, DataFrame)를 지원
# 축약 연산, 누락된 데이터 처리, Sql query, 데이터 조작, 인덱싱, 시각화.... 등 다양한 기능 제공

# Series : 일련의 데이터를 기억할 수 있는 1차원 배열과 같은 자료구조로 명시적인 색인을 갖는다.
from pandas import Series
import numpy as np

#obj = Series([3, 7, -5, 4]) # index값이 자동으로 붙음
#obj = Series([3, 7, -5, '4'])
#obj = Series([3, 7, -5, 4.5])
obj = Series((3, 7, -5, 4))
#obj = Series({3, 7, -5, 4})     # TypeError: 'set' type is unordered

print(obj, type(obj))
print()

obj2 = Series([3, 7, -5, 4], index = ['a', 'b', 'c', 'd'])  # 색인지정
print(obj2)
print(sum(obj2), np.sum(obj2), obj2.sum())  # pandas는 numpy 함수를 기본적으로 계승해서 사용

print()
print(obj2.values)
print(obj2.index)

# 슬라이싱(slicing)
print(obj2['a'], obj2[['a']])
print(obj2[['a', 'b']])
print(obj2['a':'c'])
print(obj2[2])
print(obj2[1:4])
print(obj2[[2,1]])
print(obj2 > 0)
print('a' in obj2)

print('\ndict type으로  Series를 생성')
names = {'mouse':5000,'keyboard':25000,'monitor':550000}
print(names)
obj3 = Series(names)
print(obj3, ' ', type(obj3))
print(obj3['mouse'])

obj3.name = '상품가격' # Series 객체에 이름 부여
print('\n-----------')
from pandas import DataFrame
# DataFrame : 표 모양(2차원)의 자료 구조. Series가 여러 개 합쳐진 형태. 각 칼럼마다 type이 다를 수 있다.
df = DataFrame(obj3)
print(df)
data = {
    'irum':['홍길동','한국인','신기해','공기밥','한가해'],
    'juso':('역삼동','신당동','역삼동','역삼동','신사동'),
    'nai':[23,25,33,30,35]
}
print(data, type(data))
frame = DataFrame(data) # dict로 DataFrame 생성
print(frame)
print(frame['irum'])    # 칼럼은 dict 형식으로 접근
print(frame.irum)       # 칼럼은 속성 형식으로 접근
print(frame.irum, ' ', type(frame.irum))    # 속성 형식으로 접근. Series

print()
print(DataFrame(data, columns=['juso', 'irum', 'nai'])) # 칼럼의 순서를 바꿔줌
print()
frame2 = DataFrame(data, columns=['irum', 'nai', 'juso', 'tel'], index = ['a','b','c','d','e'])
print(frame2)

frame2['tel'] = '111-1111'  # tel 칼럼에 모든 행에 적용
print(frame2)

print()
val = Series(['222-2222', '333-3333', '444-4444'], index = ['b', 'c', 'e'])
frame2['tel'] = val
print(frame2)

print()
print(frame2.T)
print(frame2.values)
print(frame2.values[0,1])
print(frame2.values[0:2])   

print('행/열 삭제')
frame3 = frame2.drop('d')   # index가 d인 행을 삭제
frame3 = frame2.drop('d', axis=0)   # index가 d인 행을 삭제(axis=0은 행단위 - 기본값임)

frame4 = frame2.drop('tel', axis=1) # tel 칼럼을 삭제
print(frame4)

print('***정렬***')
print(frame2.sort_index(axis=0, ascending=False))   # 행 단위
print(frame2.sort_index(axis=1, ascending=True))   # 열 단위
print()
print(frame2.rank(axis=0))  # 사전순으로 칼럼값 순서를 매김

print()
print(frame2['juso'].value_counts())

print('문자열 자르기')
data = {
    'juso':['강남구 역삼동','중구 신당동','강남구 대치동'],
    'inwon':[22,23,24]
}

fr = DataFrame(data)
print(fr)
result1 = Series([x.split()[0] for x in fr.juso])
result2 = Series([x.split()[1] for x in fr.juso])
print(result1)
print(result2)
print(result1.value_counts())






