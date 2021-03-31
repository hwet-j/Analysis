# file i / o
# 읽어온 파일들의 실제 파일을 실행시켜 어떻게 자료가 나눠져있는지 확인해서 비교
import pandas as pd

#df = pd.read_csv(r'../testdata/ex1.csv')
df = pd.read_csv(r'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ex1.csv', header=None)
print(df, type(df))

print()
df = pd.read_table(r'../testdata/ex2.csv', sep=',')
print(df, type(df))

print()
df = pd.read_csv(r'../testdata/ex2.csv', names=['a','b','c','d','e'])
print(df, type(df))

print()
df = pd.read_csv(r'../testdata/ex2.csv', names=['a','b','c','d','e'], index_col = 'e')
print(df, type(df))

print()
df = pd.read_csv(r'../testdata/ex3.txt', sep = '\s+', skiprows=[1,3])   
# skiprow [1,3]은 1행과 3행을 제외한 값들을 안본다.
print(df, type(df))

print()
df = pd.read_fwf(r'../testdata/data_fwt.txt', encoding='utf8', width=(10,3,5),\
                  names=('date','name','price'))
print(df, type(df))
print(df['date'])

print('--------------')
# 파일이 너무 큰 경우에는 나워서 읽기 : chunksize 옵션을 사용
test = pd.read_csv('../testdata/data_csv2.csv', header=None, chunksize = 3)
print(test)
for p in test:  # 부분 부분 끊어서 읽기 ( 위에서 지정해준 3개 씩 )
    #print(p)
    print(p.sort_values(by=2, ascending=True))
    print()

print('\n\n파일로 저장----------------')
items = {'apple':{'count':10,'price':1500},'orange':{'count':5,'price':500}}
print(items)
df = pd.DataFrame(items)
print(df)
print()
df.to_csv('result1.csv', sep=',')
df.to_csv('result2.csv', sep=',', index=False)  # 색인 제외
df.to_csv('result3.csv', sep=',', index=False, header=False)  # 색인, 칼럼명 제외

data = df.T
print(data)
data.to_html('result1.html')

# excel로 저장
print()
df2 = pd.DataFrame({'data':[1,2,3,4,5]})
wr = pd.ExcelWriter('good.xlsx', engine='xlsxwriter')
df2.to_excel(wr, sheet_name='Sheet1')
wr.save()

# excel 읽기
exf = pd.ExcelFile('good.xlsx')
print(exf.sheet_names)

dfdf = exf.parse('Sheet1')
print(dfdf)

print()
dfdf2 = pd.read_excel(open('good.xlsx', 'rb'), sheet_name='Sheet1')
print(dfdf2)










