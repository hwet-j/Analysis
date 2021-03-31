# MPL : 다층신경망 논리회로 분류 
import numpy as np
from sklearn.neural_network import MLPClassifier

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
label = np.array([0,0,0,1]) # and
# label = np.array([0,1,1,1]) # or
# label = np.array([0,1,1,0]) # xor


# verbose (진행과정을 볼수있음 기본 0, 이외 1,2값 가능 (차이점은 모름))
# max_iter 기본값 200 (학습 200번)   
# ml = MLPClassifier(hidden_layer_sizes=20).fit(feature, label)
# ml = MLPClassifier(hidden_layer_sizes=30, max_iter=150, verbose=1,learning_rate_init = 0.01).fit(feature, label)
ml = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter = 200,verbose=1,learning_rate_init = 0.01).fit(feature, label)
print(ml)
print(ml.predict(feature))
