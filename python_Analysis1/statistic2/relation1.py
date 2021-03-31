# 공분산 / 상관계수
# 공분산 : 두 개 이상의 확률변수에 대한 관게를 알려주는 값이다.값의 범위가 정해져 있지 않아 어떤 값을 기준으로 정하기 모호하다.
import numpy as np
print(np.cov(np.arange(1, 6), np.arange(2, 7))) # 2.5
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1))) # -2.5
print(np.cov(np.arange(1, 6), (3, 3, 3, 3, 3))) # 0

print(np.corrcoef(np.arange(1, 6), np.arange(2, 7)))
print(np.corrcoef(np.arange(100, 600), np.arange(200, 700)))

print()
x = [8,3,6,6,9,4,3,9,3,4]
print('x평균 : ', np.mean(x))
print('x분산 : ', np.var(x))

y = [6,2,4,6,9,5,1,8,4,5]
print('y평균 : ', np.mean(y))
print('y분산 : ', np.var(y))

# 두 변수 간의 관계 확인
print()
print('x,y 공분산 : ', np.cov(x, y)[0, 1])     # 두 변수 간에 데이터 크기에 따라 동적
print('x,y 상관계수 : ', np.corrcoef(x, y)[0, 1])   # 두 변수 간에 데이터 크기에 상관없이 값은 고정

# 피어슨 상관계수
# -1 < r < 0.7    : 강한 음의 선형관계
# -0.7 < r < -0.3 : 뚜렷한 음의 선형관계
# -0.3 < r < -0.1 : 약한 음의 선형관계
# -0.1 < r < 0.1  : 거의 무시될 수 있는 선형관계
# 0.1 < r < 0.3   : 약한 양의 상관관계
# 0.3 < r < 0.7   : 뚜렷한 양의 선형관계
# 0.7 < r < 1     : 강한 양의 선형관계

import matplotlib.pyplot as plt
plt.plot(x, y, 'o')
plt.show()

m = [-3,-2,-1,0,1,2,3]
n = [9,4,1,0,1,4,9]

plt.plot(m, n, '+')
plt.show()
print('m, n 상관계수 : ', np.corrcoef(m, n)[0, 1])









