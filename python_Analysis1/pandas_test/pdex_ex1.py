# https://cafe.daum.net/flowlife/RUrO/71
# padas 1번 문제

import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.random.randn(9, 4))
df1.columns = ['No1','No2','No3','No4']
print(df1)
print()
print(df1.mean(axis=0))