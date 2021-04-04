# 문제
# 주사위를 2회 던져서 나온 숫자들의 합이 4의 배수가 되는 경우를 출력하시오. for문 사용
for i in range(1, 7):
    for j in range(1, 7):
        if (i + j) % 4 == 0:
            print('{},{}의 숫자의 합은 4의 배수'.format(i,j))
     