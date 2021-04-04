# closure(클로저) : 내부함수의 주소를 반환해서 함수의 멤버를 계속적으로 참조

def out():
    count = 0
    def inn():
        nonlocal count
        count += 1
        return count
    
    print(inn())
    
# print(count)
out()

print()
def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
        return count
    return inner  # 이게 클로저
    
obj1 = outer() #inner의 주소를 치환
print(obj1())

result = obj1()
print(result)

print(obj1())

print('\n수랑 * 단가 * 세금 결과 출력 ----------')
def outer2(tax):
    def inner2(su, dan):
        amount = su * dan * tax
        return amount
    return inner2 # 이게 클로저 

q1 = outer2(0.1)
result1 = q1(5, 10000)
print('result1 : ', result1)

result2 = q1(10, 20000)
print('result2 : ', result2)

q2 = outer2(0.05)
result3 = q2(5, 10000)
print('result3 : ', result3)

result4 = q2(10, 20000)
print('result4 : ', result4)

print('\n재귀 함수 -------------')
def CountDown(n):
    if n == 0:
        print('처리완료')
    else:
        print(n, end = ' ')
        CountDown(n - 1) # 함수가 자신을 호출
        
CountDown(5)

print()
def totFunc(su):
    if su == 1:
        print('처리 끝')
        return True
    return su + totFunc(su - 1)

re = totFunc(10)
print('10까지의 합은 : ',re)

