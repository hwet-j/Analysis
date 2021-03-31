# RandomForestClassifier 분류 모델 연습 - 앙상블 기법(여러 개의 Decision Tree를 묶어 하나의 모델로 사용)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv")
print(df.head(3), df.shape) # (891, 12)
print(df.columns)
# print(df.info())
# print(df.isnull().any())    # na값 존재확인

df = df.dropna(subset=['Pclass','Age','Sex'])   # 사용할 데이터에서 na값 제거
print(df.head(3), df.shape) # (714, 12)

df_x = df[['Pclass','Age','Sex']]   # 사용할 데이터만 뽑아옴
print(df_x.head(3))

# female : 0 , male : 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df_x.loc[:, 'Sex'] = LabelEncoder().fit_transform(df_x['Sex'])
# df_x['Sex'] = df_x['Sex'].apply(lambda x:1 if x == 'male' else 0)
print(df_x.head(3), df_x.shape) # (714, 3)
df_y = df['Survived']
print(df_y.head(3), df_y.shape) # (714,)

df_x2 = pd.DataFrame(OneHotEncoder().fit_transform(df_x['Pclass'].values[:, np.newaxis]).toarray(),\
                     columns = ['f_class','s_class','t_class'], index=df_x.index)
# print(df_x2)
df_x = pd.concat([df_x, df_x2], axis = 1)
print(df_x.head(3))

# train/ test
(train_x, test_x, train_y, test_y) = train_test_split(df_x, df_y)

# model 
model = RandomForestClassifier(n_estimators = 500, criterion = 'entropy')
fit_model = model.fit(train_x, train_y)

pred = fit_model.predict(test_x)
print('예측값  :', pred[:10])
print('실제값  :', test_y[:10].ravel())

# 정확도 
print('acc : ', sum(test_y == pred) / len(test_y))
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(test_y, pred))




