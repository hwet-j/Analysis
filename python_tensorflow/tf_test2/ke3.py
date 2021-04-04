# cost function : cost(loss, 손실)가 최소가 되는 weight 값 찾기

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]
b = 0

# w = 1
# hypothesis = x * w + b
# cost = tf.reduce_sum(tf.pow(hypothesis - y, 2))  / len(x)

w_val = []
cost_val = []

for i in range(-50, 50):
    feed_w = i * 0.1    # 0.1 : learning rate(학습율)
    hypothesis = tf.multiply(feed_w, x) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i) + ' ' + ', cost' + str(cost.numpy()) + ', w' + str(feed_w))
    
    
plt.plot(w_val, cost_val)
plt.xlabel('w')
plt.ylabel('cost')
plt.show()
    
