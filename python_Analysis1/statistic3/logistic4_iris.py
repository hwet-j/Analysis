# LogisticRegression으로 iris의 꽃 종류를 분류

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

iris = datasets.load_iris()
print(iris.data[:3])
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3]))

x = iris.data[:, [2,3]] # 2열과 3열만 작업에 참여 (feature(독립변수, x) : Petal length, Petal width) - 편의상 2개만 
y = iris.target # label, class
print(type(x), type(y), x.shape, y.shape)
print(set(y))   # {0, 1, 2}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

# scaling (표준화 : 단위가 다른 feature가 두 개 이상인 표준화를 진행 -> 모델의 성능이 우수)
print()
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
# 표준화 값을 원래 값으로 복귀
# inver_x_train = sc.inverse_transform(x_train)
# print(inver_x_train)

# 분류 모델
# logit(), glm() : 이항분류 - 활성화 함수 : sigmoid
# LogisticRegression : 다항분류 - 활성화 함수 : softmax
model = LogisticRegression(C=1.0, random_state = 0) # C속성 : 모델에 패널티를 적용 (L2정규화) - 과적합 방지
model.fit(x_train, y_train) # 지도학습이니까

# 분류에측 
y_pred = model.predict(x_test)  # 검정자료는 test
print('예측값 : ', y_pred)
print('실제값 : ', y_test)

# 분류 정확도 
print('총 개수 : %d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print()
print('분류 정확도 출력 1: %.3f'%accuracy_score(y_test, y_pred))   # 0.956

con_mat = pd.crosstab(y_test, y_pred, rownames = ['예측치'], colnames = ['실제치'])
print(con_mat)
print('분류 정확도 출력 2:',(con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))   # 0.9555555

# 두값의 차이가 얼마나지 않아야 sampling이 잘되었다고 할수있다.
print('분류 정확도 출력3 :', model.score(x_test, y_test))  # test
print('분류 정확도 출력3 :', model.score(x_train, y_train))    # train

# 새로운 값으로 예측 
new_data = np.array([[5.1, 2.4],[1.1, 1.4],[8.1, 8.4]])
# 표준화
sc.fit(new_data)
new_data = sc.transform(new_data)
new_pred = model.predict(new_data)
print('새로운 값으로 예측 :', new_pred)


 # 붓꽃 자료에 대한 로지스틱 회귀 결과를 차트로 그리기 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc

plt.rc('font', family='malgun gothic')      
plt.rcParams['axes.unicode_minus']= False

def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')  # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])
    
    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.
    
    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)
        
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')
    
    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=model, test_idx=range(105, 150), title='scikit-learn제공') 




