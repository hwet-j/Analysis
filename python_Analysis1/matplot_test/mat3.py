# matplotlib 라이브러리 기능을 보완하기 위해 seaborn을 사용

import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset("titanic")
print(titanic.info())
print(titanic.head(3))

age = titanic["age"]

#sns.kdeplot(age)
#sns.distplot(age) # kdeplot + hist
#plt.show()

#sns.boxplot(y="age", data=titanic)
#plt.show()

#sns.relplot(x="who", y="age", data=titanic)
#plt.show()

#sns.countplot(x="class", data=titanic, hue='who')
#plt.show()

#t_pivot = titanic.pivot_table(index="class", columns="age", aggfunc="size")
#print(t_pivot)
#sns.heatmap(t_pivot, cmap=sns.light_palette('gray', as_cmap=True), \
#            annot = True)
#plt.show()

import pandas as pd
iris_data = pd.read_csv("../testdata/iris.csv")
print(iris_data.info())
print(iris_data.head(3))
'''
plt.scatter(iris_data['Sepal.Length'],iris_data['Petal.Length'])
plt.xlabel('Sepal.Length')
plt.xlabel('Petal.Length')
plt.show()

cols = []
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa': choice = 1
    elif s == 'versicolor': choice = 2
    else: choice = 3
    cols.append(choice)
    
plt.scatter(iris_data['Sepal.Length'],iris_data['Petal.Length'], c = cols)
plt.xlabel('Sepal.Length')
plt.xlabel('Petal.Length')
plt.show()
'''

# pandas의 시각화 사용
print(type(iris_data))  # <class 'pandas.core.frame.DataFrame'>
'''
from pandas.plotting import scatter_matrix
scatter_matrix(iris_data, diagonal='kde')   # pandas의 시각화 기능
plt.show()

# seaborn
sns.pairplot(iris_data, hue="Species", height = 1)
plt.show()

x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

sns.kdeplot(x)
plt.show()
'''

import numpy as np
df = pd.DataFrame(np.random.randn(10,3), \
                  index=pd.date_range('1/1/2000', periods=10), columns=['a','b','c'])
print(df)

#df.plot()
#df.plot(kind='bar')
df.plot(kind='box')
plt.xlabel('time')
plt.ylabel('data')
plt.show()

df[:5].plot.bar(rot=0)
plt.show()






