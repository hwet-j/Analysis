# 문1) 1 ~ 100 사이의 숫자 중 3의 배수이나 2의 배수가 아닌 수를 출력하고, 합을 출력
i = 0; total = 0
while i <= 100:
    i += 1
    if i % 3 == 0:
        if i % 2 != 0:
            print(i, end = ", ")
            total += i
print("총합 : " + str(total))

# 문2) 2 ~ 5 까지의 구구단 출력
i = 2
while i <= 5:
    j = 1
    while j <= 9:
         print(str(i) + 'X' + str(j) + '=' + str(int(i) * int(j)), end = "\t")
         j = j + 1
    print()
    i = i + 1

# 문3) -1, 3, -5, 7, -9, 11 ~ 99 까지의 합을 출력
i = 1 
odd = 0
even = 0
while i <= 100:
    if i % 2 != 0:
        odd = odd - i  # 홀수번째 홀수는 음수
    else:
        even = even + i # 짝수번째 홀수는 양수
    i += 1
print("총합 : " + str(odd+even))      

# 문4) 1 ~ 1000 사이의 소수(1보다 크며 1과 자신의 수 이외에는 나눌 수 없는 수)와 그 개수를 출력
num = 2
while num <= 1000:
    count = 0  # 약수의 개수를 세어줄 변수  (자신을 포함해 나눠지는수가 2개면 소수)
    i = 1  # 1 ~ num 까지 증가할 변수
    while i <= num:
        if num % i == 0:  # 나누어지면 약수
            count += 1
        i += 1  # 1증가
    if count == 2:  # 약수의 개수가 2개면 출력
        print('{0}의 약수가 {1}개이므로 "소수"입니다.'.format(num, count))
    num += 1  # 1000까지 증가
    
    









