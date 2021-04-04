# tf.constant : 텐서(상수값)를 기억, tf.Variable(텐서가 저장된 주소를 기억)

import numpy as np
import tensorflow as tf

a = 10
print(a, type(a))
print()
b = tf.constant(10)
print(b, type(b))
c = tf.Variable(10)
print(c, type(c))

print()
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
node3 = tf.add(node1, node2)
print(node3)

print()
v = tf.Variable(1)

# def find_next_odd():    # 파이썬 함수
#     v.assign(v+1)
#     if tf.equal(v % 2, 0): # 파이썬 제어문
#         v.assign(v + 10)
 
@tf.function      # 먼저 파이썬 함수로 작성을하여 오류가 나지 않을경우 사용한다. ( 속도가 더빠름 ) 
def find_next_odd():    # autograph 기능에 의해 Graph 객체 환경에서 작업할 수 있도록 코드 변환
    v.assign(v+1)
    if tf.equal(v % 2, 0): # Graph 객체 환경에서 사용하는 제어문으로 코드 변환
        v.assign(v + 10)
        
        

find_next_odd()
print(v.numpy())

print('---------------------')
def func():
    imsi = 0
#     imsi = tf.constant(0)   # imsi = 0 
    su = 1
    for _ in range(3):
        imsi += su
#         imsi = tf.add(imsi, su) # imsi += su
    return imsi

kbs = func()
print(kbs)

print('++++++++++++++++++++++')
imsi = tf.constant(0)        # 이값을 주석 해제하고 func2에있는 값을 주석처리하면 오류가 나는데 이는 for문에 있는 값이 지역변수라 존재하지 않기때문이다 .
                               # 그렇기 때문에 이렇게 작성하고 싶으면 함수 안에 global imsi를 작성하여 전역변수로 바꿔준다. 
@tf.function 
def func2():
    # imsi = tf.constant(0)
    su = 1
    global imsi
    for _ in range(3):
        imsi = tf.add(imsi, su)
        
    return imsi

mbc = func2
print(mbc())




print('=====================')
def func3():
    imsi = tf.Variable(0)
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        imsi = imsi + su    # imsi += 1 는 안됨
        
    return imsi

sbs = func3()
print(sbs)


print('~~~~~~~~~~~~~~~~~~~~~~~')
imsi = tf.Variable(0)   # autograph 밖에 선언
@tf.function
def func4():
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su    # imsi += 1 는 안됨
        imsi.assign_add(su)     # 누적은 이렇게
        
    return imsi

ytn = func4()
print(ytn)


print('------------구구단 ----------------')
@tf.function
def gugu1(dan):
    su = 0
    for _ in range(9):
        su = tf.add(su, 1)
        # print(su)
        # print(su.numpy())    # autograph 안에서 numpy()사용 불가
        # print('{} * {} = {:2}'.format(dan, su, dan * su))   # autograph 안에서 format사용 불가
        # autograph에서 서식 사용 불가 (순수하게 tensor처리만 할것)
        
print(gugu1(3))

print()
@tf.function
def gugu2(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i)
        # print(result.numpy())
        print(result)
        
print(gugu2(5))





