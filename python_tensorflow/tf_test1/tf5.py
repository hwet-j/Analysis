# 연산자 기본 함수
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = 3

# 삼항연산 
result1 = tf.cond(x > y, lambda:tf.add(x, y), lambda:tf.subtract(x, y))
print(result1, ' ', result1.numpy())

# case 조건 
f1 = lambda:tf.constant(1)
print(f1)
f2 = lambda:tf.constant(2)
print(f2())
a = tf.constant(3)
b = tf.constant(4)
result2 = tf.case([(tf.less(a, b), f1)], default = f2)
print(result2, ' ', result2.numpy())

print()
# 관계연산 
print(tf.equal(1, 2).numpy())
print(tf.not_equal(1, 2))
print(tf.less(1, 2))
print(tf.greater(1, 2))
print(tf.greater_equal(1, 2))

print()
# 논리연산
print(tf.logical_and(True, False).numpy())
print(tf.logical_or(True, False).numpy())
print(tf.logical_not(True).numpy())

print()
kbs = tf.constant([1,2,2,2,3])
val, idx = tf.unique(kbs)
print(val.numpy())
print(idx.numpy())

print()
ar = [[1,2], [3,4]]
print(tf.reduce_mean(ar).numpy())   # 차원축소를 하며 평균
print(tf.reduce_mean(ar, axis = 0).numpy())
print(tf.reduce_mean(ar, axis = 1).numpy())
print(tf.reduce_sum(ar).numpy())

print()
t = np.array([[[0,1,2], [3,4,5],[6,7,8],[9,10,11]]])
print(t.shape)
print(tf.reshape(t, shape = [2, 6]))
print(tf.reshape(t, shape = [-1, 6]))
print(tf.reshape(t, shape = [2, -1]))

print()
print(tf.squeeze(t))    # 열 요소가 1개인 경우 차원 축소
aa = np.array([[1],[2],[3],[4]])
print(aa, aa.shape)
bb = tf.squeeze(aa)
print(bb, bb.shape)

print()
print(tf.expand_dims(t, 0))
print(tf.expand_dims(t, 1))
print(tf.expand_dims(t, -1))

print()
print(tf.one_hot([0,1,2,0], depth=3))
print(tf.argmax(tf.one_hot([0,1,2,0], depth=3)).numpy())










