'''
조건 판단문 if 
'''
var = 5
if var >= 3:
    print('크구나')
    print('참일경우 실행')
    
print('계속')

if var < 3:
    print('크구나')
    print('참일경우 실행')
else:
    print('거짓일경우 실행')
    
print('계속2')

print()
money = 1000
age = 22
if money >= 500:
    item = '사과'
    if age <= 30:
        msg = '영하네'
    else:
        msg = '약간 나이가..'
else:
    item = '한라봉'
    if age >= 20:
        msg = '영한 사람이 한라봉을'
    else:
        msg = '약간 나이가 적은 사람이 한라봉을 '

print(item, msg)
print('------------------')
jumsu = 85
if jumsu >= 90:
    print('우수')
elif jumsu >= 70:
    print('일반')
else:
    print('노력 필요')

# jumsu = int(input('점수 입력 : ')) # 키보드로 받기

print()
if 90 <= jumsu <= 100:
    print('우수')
elif 70 <= jumsu < 90:
    print('일반')
else:
    print('노력필요')
    
print()
names = ['홍길동', '신기해', '김밥']
if '홍길동' in names:
    print('내친구야~')
else:
    print('넌 누구')

print()
a = 'kbs'
b = 9 if a =='kbs' else 11
print(b)

a = 11
b = 'mbc' if a == 9 else 'kbs'
print(b)

a = 3
if a > 5:
    result = a * 2
else:
    result = a + 2
print(result)

result = a * 2 if a > 5 else a + 2
print(result)

print((a + 2, a * 2)[a > 5])
# print(int(True), int(False))


