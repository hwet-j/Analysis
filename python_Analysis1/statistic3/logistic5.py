# 분류모델 성능 평가 : ROC curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

x, y = make_classification(n_samples = 16, n_features = 2, n_informative = 2, n_redundant = 0, random_state = 12)
# print(x)
print(y)

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print(y_hat)

f_value = model.decision_function(x) # 결정(판별)함수. 불확실성 추정함수 .ROC 커브로 판별 경계선 설정을 위한 sample data 제공
print(f_value)

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns = ['f', 'y_hat', 'y'])  # .T는 행렬의 위치를 바꿔줌.
df.sort_values('f',ascending=False).reset_index(drop=True)
print(df)

# ROC 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat, labels = [1,0]))
accuracy = 11 / 16
print('accuracy : ', accuracy)
recall = 6 / (6 + 3)    # 재현율, TPR
print('recall : ', recall)
fallout = 3 / (3 + 5)   # 위 양성을 FPR
print('fallout : ', fallout)

from sklearn import metrics
acc_sco = metrics.accuracy_score(y, y_hat)
cl_rep = metrics.classification_report(y, y_hat)
print('acc_sco : ', acc_sco)
print('cl_rep : ', cl_rep)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('thresholds : ', thresholds)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0,1], [0,1], 'k--', label = 'random guess')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()

# AUC (Area Under the Curve) : ROC 커브의 면적
# from sklearn.














