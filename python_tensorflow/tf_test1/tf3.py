# 탠서플로우는 탠서 계산을 그래프로 작업한다. 2.x부터는 그래프가 묵시적으로 활동한다.
# 그래프는 계산의 단위를 나타내는 tf.Operation 객체와 연산 간에 흐르는 데이터의 단위를 나타내는 tf.Tensor 객체의 세트를 포함한다.
# 데이터 구조는 tf.Graph 컨텍스트에서 정의됩니다.

import tensorflow as tf

a = tf. constant(1)
print(a)

g1 = tf.Graph()

with g1.as_default():
    c1 = tf.constant(1, name = 'c_one')
    print(c1)
    print(type(c1))
    print()
    print(c1.op)
    print()
    print(g1.as_graph_def())
    
    
print('----------------')
g2 = tf.Graph()
with g2.as_default():
    v1 = tf.Variable(initial_value=1, name = 'v1')
    print(v1)
    print(type(v1))
    print()
    print(v1.op)
    print()
    print(g2.as_graph_def())
    
    
    
    
