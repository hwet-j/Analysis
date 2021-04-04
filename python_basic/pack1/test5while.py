# 반복문 while
a = 1

while a <= 5:
    print(a, end = ' ')
    a += 1

print('\na : ', a)
print()

i = 1
while i <= 3:
    j = 1
    while j <= 4:
        print('i:' + str(i) + '/j:' + str(j))
        j = j + 1
    print('-------')
    i += 1
    
print('계속')
print('1 ~ 100 사이의 3의 배수의 합')
i = 1; hap = 0
while i <= 100:
    if i % 3 == 0:
        hap += i
    i += 1
    
print('합은' + str(hap))

print()
colors = ['r', 'g', 'b']
print(colors[1])
a = 0
while a < len(colors):
    print(colors[a], end = ' ')
    a += 1
    
print()
while colors:
    print(colors.pop(), end = ' ')
    
print('별찍기')
i = 1
while i <= 10:
    j = 1
    re = ''
    while j <= i:
        re = re + '*'
        j = j + 1
    print(re)
    i = i + 1

print()
i = 0
while i <= 10:
    j = 1
    re = ''
    while j <= i:
        re = re + ' '
        j = j + 1
    k = 1
    while k <= 11 - j:
        re = re + '*'
        k = k + 1
    print(re)
    i = i + 1

# if 블럭 내에 while 블럭
import time
sw = input('폭탄 스위치를 누를까요?[y/n]')
if sw == 'Y' or sw == 'y':
    count = 5
    while 1 <= count:
        print('%d 초 남았어요' %count)
        time.sleep(1)
        count -= 1
    print('폭발') 
elif sw == 'N' or sw == 'n':
    print('작업 취소')
else:
    print('y 또는 n을 누르시오')

