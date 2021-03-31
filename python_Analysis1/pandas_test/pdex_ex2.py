# https://cafe.daum.net/flowlife/RUrO/71
# padas 2번 문제
from pandas import Series, DataFrame

data = {
    'numbers':[10,20,30,40]
}
print(data, type(data))
frame = DataFrame(data, index = ['a','b','c','d']) 
print(frame)    # 기본
print(frame.values[2])  # c값 출력
print(frame.values[[0,3]])  # a,d값 출력
print(frame.sum())  # numbers의 합
print(frame * frame)    # 제곱한 값

frame = DataFrame(data, columns=['numbers', 'floats'], index = ['a','b','c','d'])
print(frame)    # floats 칼럼 생성
val = Series([1.5, 2.5, 3.5, 4.5], index = ['a','b','c','d'])  # floats에 넣어줄 값설정
frame['floats'] = val   # 칼럼에 값 넣어줌
print(frame)    # 출력
val2 = Series(['길동', '오정', '팔계', '오공'], index = ['d','a','b','c'])  # names에 넣어줄 값설정
print(val2)
frame['names'] = val2   # frame에 칼럼을 미리 생성하는걸 생략하고 바로 삽입
print(frame)