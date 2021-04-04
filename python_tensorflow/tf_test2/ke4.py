# GradientTape()을 이용해 최적의 w 얻기
# 경사 하강법으로 cost를 최소화

# 단순 선형회귀 예측모형 작성
# x = 5일때 f(x) = 50에 가까워지는 w값 찾기

import tensorflow as tf
import numpy as np
x = tf.Variable(5.0)
w = tf.Variable(0.0)

@tf.function
def train_step():
    with tf.GradientTape() as tape:     # 자동 미분을 위한 API 제공
        y = tf.multiply(w, x) + 0
        loss = tf.square(tf.subtract(y, 50))
    grad = tape.gradient(loss, w)  # 자동 미분
    mu = 0.01   # 학습율
    w.assign_sub(mu * grad)
    return loss

for i in range(10):
    loss = train_step()
    print("{:1}, w:{:4.3}, loss:{:4.5}".format(i, w.numpy(), loss.numpy()))
    
print('\n옵티마이저 객체를 사용')
opti = tf.keras.optimizers.SGD()

x = tf.Variable(5.0)
y = tf.Variable(0.0)

@tf.function
def train_step2():
    with tf.GradientTape() as tape:     # 자동 미분을 위한 API 제공
        y = tf.multiply(w, x) + 0
        loss = tf.square(tf.subtract(y, 50))
    grad = tape.gradient(loss, w)  # 자동 미분
    opti.apply_gradients([(grad, w)])
    return loss

for i in range(10):
    loss = train_step2()
    print("{:1}, w:{:4.3}, loss:{:4.5}".format(i, w.numpy(), loss.numpy()))
    
print('\n최적의 기울기, 절편 구하기 ---------------------')
opti = tf.keras.optimizers.SGD()
x = tf.Variable(5.0)
w = tf.Variable(0.0)
b = tf.Variable(0.0)

@tf.function
def train_step3():
    with tf.GradientTape() as tape:     # 자동 미분을 위한 API 제공
        #y = tf.multiply(w, x) + 0
        y = tf.add(tf.multiply(w, x), b)
        loss = tf.square(tf.subtract(y, 50))
    grad = tape.gradient(loss, [w, b])  # 자동 미분
    opti.apply_gradients(zip(grad, [w, b]))
    return loss

w_val = []  # 시각화를 목적으로.....
cost_val = []

for i in range(10):
    loss = train_step3()
    print("{:1}, w:{:4.3}, loss:{:4.5}, b:{:4.3}".format(i, w.numpy(), loss.numpy(), b.numpy()))
    w_val.append(w.numpy())
    cost_val.append(loss.numpy())
    
import matplotlib.pyplot as plt
plt.plot(w_val, cost_val, 'o')
plt.xlabel('W')
plt.ylabel('Cost')
plt.show()

print('\n선형회귀 모형 작성 ----------------------')
opti = tf.keras.optimizers.SGD()

w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))


@tf.function
def train_step4(x, y):
    with tf.GradientTape() as tape: # 자동 미분을 위한 API 제공
        hypo = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
    grad = tape.gradient(loss, [w, b])   # 자동 미분
    opti.apply_gradients(zip(grad, [w, b]))
    return loss

x = [1.,2.,3.,4.,5.]    # feature
y = [1.2, 2.0, 3.0, 3.5, 5.5]   # label 
    
w_vals = []
loss_vals = []

for i in range(100):
    loss_val = train_step4(x, y)
    loss_vals.append(loss_val.numpy())
    if i % 10 == 0:
        print(loss_val)
    w_vals.append(w.numpy())
    
print('loss_vals :', loss_vals)
print('w_vals :', w_vals)

plt.plot(w_vals, loss_vals, 'o--')
plt.xlabel('W')
plt.ylabel('Cost')
plt.show() 
    
    
# 선형회귀선 시각화
y_pred = tf.multiply(x, w) + b  # 모델이 완성됨
print('y_pred :', y_pred.numpy())

plt.plot(x, y, 'ro')
plt.plot(x, y_pred, 'b--')
plt.show()

