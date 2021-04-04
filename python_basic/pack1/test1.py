# 주석, 변수, 타입 (list, tuple, set, dict)
# https://velog.io/@inyong_pang/Python-List-Tuple-Dictionary-and-Set-%EC%9A%94%EC%95%BD

# 파이썬 한줄 주석
'''
파이썬 여러줄 주석1
파이썬 여러줄 주석 1
파이썬 여러줄 주석 1
'''

"""
파이썬 여러줄 주석 2
파이썬 여러줄 주석 2
파이썬 여러줄 주석 2
"""

from future.builtins.misc import isinstance

print('파이썬 세상에 오신걸 환영합니다.')
var1 = '안녕 파이썬!'
print(var1) # 안녕 파이썬!
var1 = 10   # 같은 변수명으로 변수를 지정하면 나중 값으로 지정됨
print(var1) # 10

a = 1
b = 1.2
c = b
print(a,b,c)
print(id(a), id(1), id(b), id(c)) #주소 확인함수 id()
# 파이썬은 기본형이 존재하지 않음. 참초형은 존재함.
print(a is b, a == b)  #is 주소비교, == 값 비교
# False False
print(b is c, b == c)
# True True


A = 1
a = 2
# 변수는 대소문자를 구분한다.
print(A, a, A == a)

# for = 5 # 변수명으로 키워드(예약어) 불가
import keyword #외부모듈 읽기 - 설치된 모듈 로딩
print(keyword.kwlist) # ['False', 'None', 'True', 'and', 'as', 'assert', 'async'......

print(10, oct(10), hex(10), bin(10)) # 10 0o12 0xa 0b1010  
# oct - 8진수, hex - 16진수, bin - 2진수 (앞에 2자는 몇진수인지 알려줌)
print(10, 0o12, 0xa, 0b1010)

print('type 확인')
print(3, type(3))  # 3 <class 'int'>
print(3.4, type(3.4))  # 3.4 <class 'float'>
print(3 + 4j, type(3 + 4j))  # (3+4j) <class 'complex'>
print(True, type(True))  #  True <class 'bool'>
print('aa', type('aa'))  #  aa <class 'str'>
print("abc", type("abc"))  #  abc <class 'str'>

print(isinstance(1, int))  # isinstance() 객체 type 확인 함수 
print(isinstance(1.2, int))


print('\n') # console 줄바꿈
print('-----집합형 자료형 ------')  # 알아둬아함 ! 
print((1), type((1)))  # int 집합형 자료형 아님 - 다른것과 비교하기위해 작성해놓음

print([1], type([1]))  # list
# 리스트는 대괄호[]로 표현하며 내부의 element들의 타입이 달라도(integet, string) 표현이 가능하다.
# 인덱싱(순서)을 통해 element를 불러올수있고 생성,삭제,수정이 가능하다. 마지막 index(시작은 0부터)는 -1
# list안에 list선언가능  Ex) a = [1, 2, 3, ['a', 'b', 'c']]
# list안에 list를 불러올때 a[-1][1] ==> b, a[0:2] ==> 1, 2 
# 리스트의 연산 
# a = [1,2,3] , b = [4,5,6]
# (+)         a + b ==> [1,2,3,4,5,6]
# (*)         a * 3 ==> [1,2,3,1,2,3,1,2,3]
# 수정 : a[2] = 4
# 삭제 : del a[1]
#      remove(3) - 리스트에서 처음으로 나온 3값 하나 삭제
# 추가 : a.append(4) - 리스트의 맨 마지막에 추가됨
#      a.insert(0,4) - 리스트의 0번째(index) 자리에 4(value)를 추가
# pop() : a = [1,2,3,4,5]  
#         a.pop(2) --> 3 뽑아내(index값) list내부에서 그값 삭제 - print(a.pop(2))로 출력가능
#         print(a) --> [1,2,4,5]

# count(원소) : 리스트 내 특정 원소가 몇 개 포함되어 있는지 반환
# index(원소) : 리스트 내 특정 원소의 인덱스를 반환
# append(원소) :리스트의 뒤쪽에 새로운 원소를 삽입
# sort() : 리스트를 오름차순으로 정렬
# extend(리스트) : 리스트의 뒤쪽에 다른 리스트를 삽입
# insert(인덱스, 원소) : 특정한 위치(인덱스)에 원소를 삽입
# remove(원소) : 리스트 내 특정 원소를 삭제
# pop(인덱스) : 리스트 내 특정 인덱스의 원소를 삭제
# reverse() : 리스트의 순서를 뒤집기

print((1,), type((1,)))  # tuple
# Tuple은 element들을 '()'로 감싸고 있어 List와 비슷한 역할을 하지만, 다른 특성을 가지고 있다.
# index(순서)가 존재하지만 element들의 생성, 삭제, 수정이 불가능하다.
# List의 +, * 는 동일하게 사용함

print({'k':1}, type({'k':1}))  # dict
# 데이터들의 대응관계(속성과 값)를 잘 나타낼 수 있는 자료형이 딕셔너리(Dictionary) 
# {Key1:Value1, Key2:Value2, Key3:Value3, ...} Value에 리스트도 넣을 수 있다
# a = {1: 'a', 2: 'b', 'name': 'pey', 3: [1, 2, 3]}
# a['age'] = [1,2,3]  # a[key] = value   / key값이 기존에 존재하지않으면 a 맨뒤에 key value값이 추가됨
# print(a)     # {1: 'a', 2: 'b', 'name': 'pey', 3: [1, 2, 3], 'age': [1, 2, 3]}
# a.get('name') , a['name'] 같은 의미 작성해준key의 value를 가져옴(key가 존재하지 않을때 get은 None값을 ['']로 선언한값은 오류를 발생시킴 
# a.keys() >> key값들만 불러옴 
# a.value() >> value 값들만 불러옴
# item() >> key,value를 묶은 값들을 불러옴 

print({1}, type({1}))  # set
# 집합에 관련된 것을 쉽게 처리하기 위해 만든 자료형 (중복을 허용하지 않는다, 순서가 없다)
# 교집합 '&', intersection()
# s1 & s2 , s1.intersection(s2)
# 합집합 '|', union()
# s1 | s2 , s1.union(s2)
# 차집합 '-', difference()
# s1 - s2 , s1.difference(s2)
# set 자료형에 값 추가
# s1.add(4) - add는 1개의 값만 추가가능
# s1.update([4, 5, 6]) - update로 여러개의 값을 추 가
# s1.remove(2) - 특정 값 제거

