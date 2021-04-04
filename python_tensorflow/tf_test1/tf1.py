import tensorflow as tf

print(tf.__version__)
# print("GPU 사용 가능" if tf.test.is_gpu_available() else '사용불가')

# 상수
print(1, type(1))
print(tf.constant(1), type(tf.constant(1))) # scala 0-D tensor
print(tf.constant([1])) # vector 1-D tensor
print(tf.constant([[1]]))   # matrix 2-D tensor

print()
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c)
c = tf.add(a, b)
print(c)
d = tf.constant([3]) 
e = c + d
print(e)

print()
print(7)
print(tf.convert_to_tensor(7, dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))

print('\nnumpy의 narray와 tensor 사이의 형변환 -------')
import numpy as np
arr = np.array([1, 2])
print(arr, type(arr))
tfarr = tf.add(arr, 5)  # ndarray가 tensor type으로 자동 형변환
print(tfarr)
print(tfarr.numpy())    # tensor가 ndarray type으로 강제 형변환
print(np.add(tfarr, 3)) # tensor가 ndarray type으로 자동 형변환


