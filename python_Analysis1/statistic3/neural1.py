# 단층 신경망(뉴런, Node) - 입력자료 각각에 가중치를 곱해 더한값을 대상으로 임계값(활성화함수)을 기준하여 이항분류가 가능. 예측도 가능.
# 단층 신경망으로 논리회로 분류

def or_func(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    sigma = w1 * x1 + w2 * x2 + 0
    if sigma <= theta:
        return 0
    elif sigma > theta:
        return 1
    
print(or_func(0, 0))
print(or_func(1, 0))
print(or_func(0, 1))
print(or_func(1, 1))

print('---------------------------')
def and_func(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    sigma = w1 * x1 + w2 * x2 + 0
    if sigma <= theta:
        return 0
    elif sigma > theta:
        return 1
    
print(and_func(0, 0))
print(and_func(1, 0))
print(and_func(0, 1))
print(and_func(1, 1))


print('---------------------------')
def nand_func(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    sigma = w1 * x1 + w2 * x2 + 0
    if sigma <= theta:
        return 0
    elif sigma > theta:
        return 1
    
print(nand_func(0, 0))
print(nand_func(1, 0))
print(nand_func(0, 1))
print(nand_func(1, 1))


print('---------------------------')
def xor_func(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    sigma = w1 * x1 + w2 * x2 + 0
    if sigma <= theta:
        return 0
    elif sigma > theta:
        return 1
    
print(xor_func(0, 0))
print(xor_func(1, 0))
print(xor_func(0, 1))
print(xor_func(1, 1))

print('---------------------------')
import numpy as np
from sklearn.linear_model import Perceptron

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1]) # and
# label = np.array([0,1,1,1]) # or
label = np.array([0,1,1,0]) # xor

ml = Perceptron(max_iter = 100000).fit(feature, label)    # max_iter 학습 횟수
print(ml)
print(ml.predict(feature))




