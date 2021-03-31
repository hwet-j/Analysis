# 구조 : stack, unstack, cut, merge, concat, pivot....
import pandas as pd
import numpy as np

df = pd.DataFrame(1000 + np.arange(6).reshape(2, 3), index=['대전','서울'], columns = ['2019','2020','2021'])
print(df)
print()
#print(df.info())

print()
df_row = df.stack()
print(df_row)
df_col = df_row.unstack()
print(df_col)

# 범주화
# Cut
print()
price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11] # 3초과 7이하, 7초과 9이하, 9초과 11이하
result_cut = pd.cut(price, cut) # 정해진 cut범주 안에 price가 속한 개수
print(result_cut)
print(pd.value_counts(result_cut))

print()
datas = pd.Series(np.arange(1, 1001))
print(datas.head(3))
print(datas.tail(3))
result_cut2 = pd.cut(datas, 3)
print(result_cut2)
print(pd.value_counts(result_cut2))

# Merge
print()
df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','a','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'key':['a','b','d'], 'data2':range(3)}) # merge를 해주기위해서는 공통 컬럼 필요 (key가 공통임)
print(df2)

print()
print(pd.merge(df1, df2))
print(pd.merge(df1, df2, on = 'key'))   # inner join

print()
print(pd.merge(df1, df2, on = 'key', how = 'inner'))   # inner join

print()
print(pd.merge(df1, df2, on = 'key', how = 'outer'))   # full outer join

print()
print(pd.merge(df1, df2, on = 'key', how = 'left'))   # left outer join

print()
print(pd.merge(df1, df2, on = 'key', how = 'right'))   # right outer join

print('\n공통 칼럼명이 없는 경우')
df3 = pd.DataFrame({'key2':['a','b','d'], 'data2':range(3)})
print(df3)
print(df1)
print(pd.merge(df1, df3, left_on='key', right_on='key2'))

print()
print(pd.concat([df1, df3]))
print(pd.concat([df1, df3], axis=0))    # 열 단위로 처리 : default
print(pd.concat([df1, df3], axis=1))    # 행 단위로 처리

print('\n피봇(pivot) 테이블: 데이터의 행렬을 재구성 하여 그룹화 처리')
data = {'city':['강남','강북','강남','강북'],
        'year':[2000,2001,2002,2002],
        'pop':[3.3, 2.5, 3.0, 2]
}
df = pd.DataFrame(data)
print(df)

print(df.pivot('city','year','pop'))    # city, year별  pop의 평균
print(df.set_index(['city','year']).unstack())  # 기존의 행 인덱스를 제거하고 첫번째 열 인덱스 설정

print()
print(df.pivot('year','city','pop'))

print(df['pop'].describe())

print('--------------')
hap = df.groupby(['city'])
print(hap.sum())
print(df.groupby(['city']).sum())
print(df.groupby(['city', 'year']).sum())
print(df.groupby(['city', 'year']).mean())
print('****')

print('**pivot, groupby 중간적 성격 : pivot_table **')
print(df)
print(df.pivot_table(index=['city']))   # aggfunc=np.mean  : default
print(df.pivot_table(index=['city'], aggfunc=np.mean))
print()
print(df.pivot_table(index=['city','year'], aggfunc=[len, np.sum]))
print()
print(df.pivot_table(values=['pop'], index='city')) # city별 pop의 평균
print(df.pivot_table(values=['pop'], index='city', aggfunc=np.mean))
print(df.pivot_table(values=['pop'], index='city', aggfunc=len))

print()
print(df.pivot_table(values=['pop'], index=['year'], columns=['city']))

print()
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], margins=True))
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], margins=True, fill_value=0))













