# 반복문 while : continue, break
a = 0

while a < 10:
    a += 1
    if a == 5:continue  # continue는 이전까지의 문장을 실행하고 이후의 문장을 무시하고 다음 조건의 while문 실행
    if a == 7:break # break 조건에 맞으면 이전까지만 실행후 while자체를 나감
    print(a)
else:  # while문이 정상적으로 끝이 났는지 (break문을 만났는지) 
    print('while 정상 수행')    
print('while 수행 후%d'%a)

# 홀수, 짝수 확인
# while 1:  # 과 같이 무한루프 돌릴수있다. ( 0을 작성하면 안됨 )
while True:
    a = int(input('숫자 입력:'))
    if a == 0:
        print('안녕~')
        break
    elif a % 2 == 0:
        print('%d는 짝수'%a)
        continue
    elif a % 2 == 1:
        print('%d는 홀수'%a)
        continue


