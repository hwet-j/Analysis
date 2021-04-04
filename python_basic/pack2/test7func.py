# 함수(function) : 여러 개의 명령문을 하나의 묶음으로 만든 실항단위

# 내장함수 
print(sum([3, 5, 7]))
print(bin(8))
print(int(1.7), float(7), str(5) + '오')

a = 10
b = eval('a + 5')
print(b)

print(round(1.2), round(1.7))

import math
print(math.ceil(1.2), math.ceil(1.7))
print(math.floor(1.2), math.floor(1.7))

print()
x = [10, 20, 30]
y = ['a', 'b']
for i in zip(x, y):
    print(i)
    
# 사용자 함수
def DoFunc1():
    print('사용자 정의 함수1 : DoFunc')
    
def DoFunc2(name):
    print('안녕', name)
    
print('뭔가를하다가')
DoFunc1()
print('뭔가를하다가2')
DoFunc1()
print(DoFunc1) # 함수 이름은 객체(함수 본체)의 주소를 기억
print(type(DoFunc1))
otherFunc = DoFunc1 # 주소 치환
otherFunc()
    
print(globals())
print()
DoFunc2('tom')
# DoFunc2('tom', 'james') err
DoFunc2(100)     

def DoFunc3(ar1, ar2):
    imsi = ar1 + ar2
    print('imsi :',imsi)
    # return None 기본값
    # return imsi
    a = 1
    if a % 2 ==1:
        return
    else:
        return a
    
    print('죽은 문장')

DoFunc3(10, 20)
DoFunc3('kbs', 'mbc')
print(DoFunc3('kbs', 'mbc'))

print()
def area_tri(a,b):
    c = a * b / 2
    if a == 0: return # 함수의 무조건 탈출
    area_print(c) # 함수는 함수 호출 가능
    
def area_print(c):
    print('삼각형의 면적은 ', c)

def abc():
    pass

area_tri(20, 30)

def exam(a, b):
    ss = str(a) + '+' + str(b) + '=의 답은 :'
    ans = input(ss)
    return a + b == int(ans)

if(exam(5, 2)):
    print('나이스')
else:
    print('아쉽')

print()
def swap(a, b):
    return b, a

a = 10; b = 20
print(swap(a, b))

print()
def isOdd(arg):
    return arg % 2 == 1

print(isOdd(3))
print(isOdd(4))

myDict = {x:x*x for x in range(11) if isOdd(x)}
print(myDict)