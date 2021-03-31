# 차트의 종류 경험
import numpy as np
import matplotlib.pyplot as plt

###################
'''
fig = plt.figure()  # 명시적으로 차트 영역 객체 선언
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(np.random.randn(10), bins=10, alpha=0.1)
ax2.plot(np.random.randn(10))
plt.show()
'''

######################
'''
fig, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(np.random.randn(10))
ax[1].plot(np.random.randn(10) + 20)
plt.show()
'''

#########################
'''
data = [50, 80, 100, 70, 90]
plt.bar(range(len(data)), data) # bar대신 barh로작성해주면 가로로 되어있는 그래프
plt.show()

data = [50, 80, 100, 70, 90]
err = np.random.randn(len(data))
plt.barh(range(len(data)), data, xerr=err, alpha=0.6)
plt.show()
'''

##############################
'''
data = [50, 80, 100, 70, 90]
plt.pie(data, explode=(0, 0.2, 0, 0, 0), colors = ['yellow', 'blue', 'red'])
plt.show()
'''

####################
n = 30
np.random.seed(42)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (15 * np.random.rand(n)) ** 2
plt.scatter(x, y, s = scale, c = color)
plt.show()

import pandas as pd 
sdata = pd.Series(np.random.rand(10).cumsum(), index = np.arange(0, 100, 10))
plt.plot(sdata)
plt.show()

##############
fdata = pd.DataFrame(np.random.rand(1000, 4), index = pd.date_range('1/1/2000', periods = 1000),\
                     columns = list('ABCD'))
print(fdata)
fdata = fdata.cumsum()
plt.plot(fdata)
plt.show()


