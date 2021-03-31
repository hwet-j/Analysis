# 분산(표준편차)의 중요함 - 분포를 알 수 있기 때문

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

centers = [1, 1.5, 2]
col = 'rgb'
data = []
std = 0.01  # 표준편차 임의설정

for i in range(3):
    data.append(stats.norm(centers[i], std).rvs(100))   # rvs 는 샘플링 100개
    # 표준편차가 (std)인 랜덤한 값 샘플 100개
    plt.scatter(np.arange(100) + i * 100, data[i], color = col[i])

print(data)
plt.show()
    





















