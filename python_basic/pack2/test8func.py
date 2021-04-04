# 변수의 생존 범위 Local > Enclosing function > Global

player = '전국대표'     # Global variable

def funcSoccer():
    player = '지역대표' # Local variable
    name = '한국인'    # Local variable
    print(name, player)

funcSoccer()    
# print(name,player)

print()
a = 10; b = 20; c = 30
print('1) a:{}, b:{}, c:{}'.format(a,b,c))

def Foo():
    a = 40
    b = 50
    def Bar():
        global c # c의 값을 global변수로 바꿔줌 
        nonlocal b
        print('2) a:{}, b:{}, c:{}'.format(a,b,c))
        # global 없이 c 선언시 오류.
        c = 60  # UnboundLocalError: local variable 'c' referenced before assignment
        b= 70
    Bar()
    a = 80

Foo()
print('3) a:{}, b:{}, c:{}'.format(a,b,c))

print('***' * 10)
# argument 키워드로 매칭
def ShowGugu(start = 1, end = 5):
    print(start, end)

ShowGugu(2, 8)
ShowGugu(start = 2, end = 8)
ShowGugu()
ShowGugu(2)
ShowGugu(start = 2)
ShowGugu(end = 10)

# ShowGugu(start = 2, 7)
ShowGugu(2,end = 7)

print('***' * 10)
# 가변인수 : 인수의 개수가 부정확한 경우
def func1(*ar):
    print(ar)
    for i in ar:
        print('음식: ' + i)

func1('아아')
func1('비빔밥', '공기밥', '김밥')
print()
# def func2(*ar, a): # err
def func2(a, *ar):
    print(a)
    print(ar)
    for i in ar:
        print('음식: ' + i)

func2('비빔밥', '공기밥', '김밥')

print()
def selectProcess(choice, *ar):
    if choice == '+':
        re = 0
        for i in ar:
            re += i
    elif choice == '*':
        re = 1
        for i in ar:
            re *= i
    return re

print(selectProcess('+', 1,2,3,4,5))
print(selectProcess('*', 1,2,3,4,5))

print()
def func3(w, h, **etc):
    print('몸무게 {}, 키 {}'.format(w, h))
    print(etc)
    
func3(65, 175, irum='홍길동')
func3(65, 179, irum='고길동', age=22)
func3(w = 80, h = 170, irum='신기해')

print()
def func4(a, b, *v1, **v2):
    print(a, b)
    print(v1)
    print(v2)
    
func4(1, 2)
func4(1, 2, 3, 4, 5)
func4(1, 2, 3, 4, 5, kbs = 9, sbs = 5)

