# 회귀분석 모델 : 자동차 연비 예측

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

dataset = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/auto-mpg.csv")
print(dataset.head(3))
del dataset['car name']
print(dataset.head(3))
pd.set_option('display.max_columns', 100)
print(dataset.corr())
dataset.drop(['cylinders', 'acceleration', 'model year','origin'], axis='columns', inplace=True)
print()
print(dataset.head(2))
dataset['horsepower'] = dataset['horsepower'].apply(pd.to_numeric, errors='coerce') # data중에 ? 가 있어 형변환 하면 NaN이 생김
print(dataset.info())
print(dataset.isna().sum()) # 6
dataset = dataset.dropna()

import seaborn as sns
# sns.pairplot(dataset[['mpg','displacement','horsepower','weight']], diag_kind='kde')
# plt.show()

# train / test
train_dataset = dataset.sample(frac=0.7, random_state=123)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape)  # (274, 4)
print(test_dataset.shape)   # (118, 4)

# 표준화 작업(수식을 직접 사용)을 위한 준비
train_stat = train_dataset.describe()
# print(train_stat)
train_stat.pop('mpg')
train_stat = train_stat.transpose()
print(train_stat)

# label : mpg
train_labels = train_dataset.pop('mpg')
print(train_labels[:2])
test_labels = test_dataset.pop('mpg')
print(test_labels[:2])

def st_func(x):
    return ((x - train_stat['mean']) / train_stat['std'])

# print(st_func(10))
# print(train_dataset[:3])
# print(st_func(train_dataset[:3]))
st_train_data = st_func(train_dataset)  # train feature
st_test_data = st_func(test_dataset)  # test feature
# ------------ 모델에 적용할 dataset 준비 완료 -------------------

# Model
def build_model():
    network = tf.keras.Sequential([
        layers.Dense(units=64, input_shape = [3], activation='linear'),
        layers.Dense(64, activation='linear'),  # relu
        layers.Dense(1, activation='linear'),
    ])
    # opti = tf.keras.optimizers.RMSprop(0.01)
    opti = tf.keras.optimizers.Adam(0.01)
    network.compile(optimizer = opti, loss='mean_squared_error',\
                     metrics=['mean_absolute_error','mean_squared_error'])
    return network
    
print(build_model().summary())
model = build_model()
# fit() 전에 모델을 실행해 볼 수도 있다.
print(model.predict(st_train_data[:1]))

# 훈련
epochs = 5000

# 학습 조기 종료 
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(st_train_data, train_labels, batch_size=32,\
                    epochs=epochs, validation_split=0.2, verbose=1)
df = pd.DataFrame(history.history)
print(df.head(3))
print(df.columns)

# 시각화
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize = (8, 12))
    
    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.ylabel('Mean Abs Error[MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='train error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='val error')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.ylabel('Mean Abs Error[MPG]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='train error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='val error')
    plt.legend()
    plt.show()
    
plot_history(history)
    










