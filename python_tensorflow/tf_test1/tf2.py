# 변수 
import tensorflow as tf

print(tf.constant([1]))

f = tf.Variable(1)
print(f)
v = tf.Variable(tf.ones(2,))    # 1-D
m = tf.Variable(tf.ones(2,1))    # 2-D
print(v, m)
print(m.numpy())

print()
v1 = tf.Variable(1)
print(v1)
v1.assign(10)
# v1 = 10 # 그냥 상수 10..
print(v1, type(v1))

print()
v2 = tf.Variable(tf.ones(shape=(1)))
v2.assign([20])
print(v2)

v3 = tf.Variable(tf.ones(shape=(1,2)))
v3.assign([[30, 40]])
print(v3)

print()
v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
print(v3)

print()
var = tf.Variable([1,2,3,4,5], dtype=tf.float64)
result1 = var + 10
print(result1)

print()
w = tf.Variable(tf.ones(shape=(1,)))
b = tf.Variable(tf.ones(shape=(1,)))
w.assign([2])
b.assign([2])

def func1(x):    # 파이썬 함수
    return w * x + b

print(func1(3))
print(func1([3]))
print(func1([[3]]))

print()
@tf.function # auto graph 기능 : tf.Graph + tf.Session 파이썬 함수를 호출 가능한 그래프 객체로 변환. 텐서플로우 그래프에 포함 되어 실행됨
def func2(x):
    return w * x + b

print(func2(3))

print('------------------')
w = tf.Variable(tf.keras.backend.random_normal([5, 5], mean=0, stddev=0.3))
print(w.numpy())
print(w.numpy().mean)
print(w.numpy(w.numpy()))
b = tf.Variable(tf.zeros([5]))
print(w + b)

# assign
print()
aa = tf.ones((2,1))
print(aa.numpy())
m = tf.Variable(tf.zeros((2,1)))
m.assign(aa)
print(m.numpy())
print()

m.assign_add(aa)
print(m.numpy)

m.assign_sub(aa)
print(m.numpy)

print()
m.assign(2 * m)
print(m.numpy())




