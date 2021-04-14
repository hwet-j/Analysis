# pandas 문제 1)
#   a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오
#   b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No2, No4로 지정하시오
#   c) 각 컬럼의 평균을 구하시오

import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(9, 4))
df1.columns = ['No1', 'No2', 'No3', 'No4']
print(df1)
print()
print(df1.mean(axis = 0))

print('-----------------------------')
# pandas 문제 2)
# a) DataFrame으로 위와 같은 자료를 만드시오. 컬럼 이름은 numbers, 로우 네임은 a~d, 값은 10~40
# b) c 로우의 값을 가져오시오.
# c) a, d 로우들의 값을 가져오시오.
# d) numbers의 합을 구하시오.
# e) numbers의 값들을 각각 제곱하시오 아래 결과가 나와야 함.

df2 = pd.DataFrame([[i] for i in range(10, 41, 10)],
                   columns = ['numbers'],
                   index = ['a', 'b', 'c', 'd'])
print('a)\n', df2)
print('b)\n', df2.loc['c'])
print('c)\n', df2.loc[['a', 'd']])
print('d)\n', df2.numbers.sum())
print('e)\n', df2 ** 2)

df2['floats'] = [1.5, 2.5, 3.5, 4.5]
print('f)\n', df2)

df2['names'] = pd.Series(['길동', '오정', '팔계', '오공'], index = ['d', 'a', 'b', 'c'])
print('g)\n', df2)
