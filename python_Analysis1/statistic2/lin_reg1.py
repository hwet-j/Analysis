# 최소제곱해를 선형 행렬 방정식으로 얻기

import numpy.linalg as lin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
plt.plot(x, y)
plt.grid(True)
plt.show()

A = np.vstack([x, np.ones(len(x))]).T
print(A)

# y = mx + c 
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, ' ', c)    # 기울기 : 0.9999999999999999  절편 : -0.9499999999999997

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
plt.show()

# yhat = 0.9999999999999999 * x + -0.9499999999999997
print(0.9999999999999999 * 1 + -0.9499999999999997)
print(0.9999999999999999 * 3 + -0.9499999999999997)
print(0.9999999999999999 * 123 + -0.9499999999999997)





