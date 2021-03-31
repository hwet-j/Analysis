# pandas 문제 4)
import pandas as pd

data = pd.read_csv(r'../testdata/human.csv')
print(data)
df = pd.DataFrame(data)
print(df)
print(df[' Group'])
df_d = df.drop(df.loc[df[' Group']=='NA'].index, inplace=True)
print(df_d)

