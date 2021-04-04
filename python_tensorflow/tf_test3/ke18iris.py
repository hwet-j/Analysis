'''
# iris dataset으로 분류 모델 작성 후 ROC curve 까지 출력
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from _tkinter import create

iris = load_iris()
# print(iris.DESCR)

x = iris.data
print(x[:2])
y = iris.target
print(set(y))

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label 원핫 인코딩
one_hot = OneHotEncoder()   # to_categorical, ..........
y = one_hot.fit_transform(y[:, np.newaxis]).toarray()
print(y[:2])

# feature 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train / test
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=1)
n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features, n_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# n의 개수 만큰 모델 생성 함수
def create_custom_model(input_dim, output_dim, out_nodes, n, model_name='model'):
    def create_model():
        model = Sequential(name = model_name)
        for _ in range(n):
            model.add(Dense(out_nodes, input_dim = input_dim, activation = 'relu'))
            
        model.add(Dense(output_dim, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 10, n, 'model{}'.format(n)) for n in range(1, 4)]

for create_model in models:
    print('---------------------')
    create_model().summary()

history_dict = {}
# train 
for crate_model in models:
    model = create_model()
    print('Model name : ', model.name)
    history = model.fit(x_train, y_train, batch_size=5, epochs=50, verbose=0, validation_split=0.3)
    score = model.evaluate(x_test, y_test)
    print('test dataset loss : ', score[0])
    print('test dataset acc : ', score[1])
    history_dict[model.name] = [history, model]
    

print(history_dict)

# 시각화 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 6))
print(fig, ax1, ax2)

for model_name in history_dict:
    print('h_d :', history_dict[model_name][0].history['acc'])
    
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation acc')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
    
plt.show()

# 분류모델에 대한 성능 평가 : ROC curve
plt.figure()
plt.plot([0,1], [0,1], 'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label='{}, AUC value : {:.3}'.format(model_name, auc(fpr, tpr)))
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# k - fold 교차 검증
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

create_model = create_custom_model(n_features, n_classes, 10, 3)
estimator = KerasClassifier(build_fn=crate_model, epochs=50, batch_size=10, verbose=2)
scores = cross_val_score(estimator, x_scale, y, cv = 10)
print('accuracy : {:0.2f}(+/-{:0.2f})'.format(scores.mean(), scores.std()))

# 모델3의 성능이 가장 우수
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=2)
print(model.evaluate(x_test, y_test))

y_pred = np.argmax(model.predict(x_test), axis=1)
print('예측값 : ', y_pred)
real_y = np.argmax(y_test, axis = 1).reshape(-1,1)
print('실제값 : ', real_y.ravel())
print('분류 실패 수 : ', (y_pred != real_y.ravel()).sum())

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(real_y, y_pred))
print(accuracy_score(real_y, y_pred))
print(classification_report(real_y, y_pred))

# 새로운 값으로 예측
new_x = [[5.5, 3.3, 1.2, 1.3],[3.5, 3.3, 0.2, 0.3], [1.5, 1.3, 6.2, 6.3]]
new_x = StandardScaler().fit_transform(new_x)
new_pred = model.pred(new_x)
print('예측값 : ', np.argmax(new_pred, axis=1). reshape(-1,1).flatten())
'''
#####################################################################
#####################################################################
#####################################################################
#####################################################################

# iris dataset으로 종류별로 분류 모델 작성 후 ROC curve 까지 출력
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

iris = load_iris()
# print(iris.DESCR)

x = iris.data
print(x[:2])
y = iris.target
print(set(y))

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label 원핫인코딩
one_hot = OneHotEncoder()   # to_categorical, ...
y = one_hot.fit_transform(y[:, np.newaxis]).toarray()
print(y[:2])

# feature 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train/test
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.3, random_state = 1)
n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features, n_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# n의 갯수 만큼 모델 생성 함수
def create_custom_model(input_dim, output_dim, out_nodes, n, model_name = 'model'):
    def create_model():
        model = Sequential(name = model_name)
        for _ in range(n):
            model.add(Dense(out_nodes, input_dim = input_dim, activation = 'relu'))
            
        model.add(Dense(output_dim, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1,4)]

for create_model in models:
    print('------')
    create_model().summary()

history_dict = {}
# train
for create_model in models:
    model = create_model()
    print('Model name : ', model.name)
    history = model.fit(x_train, y_train, batch_size = 5, epochs = 50, verbose = 0, validation_split=0.3)
    score = model.evaluate(x_test, y_test)
    print('test dataset loss : ', score[0])
    print('test dataset acc : ', score[1])
    history_dict[model.name] = [history, model]
print(history_dict)

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,6))
print(fig, ax1, ax2)

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['acc'])
    
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('validation acc')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()

# 분류모델에 대한 성능 평가 : ROC curve
plt.figure()
plt.plot([0,1], [0,1],'k--')

from sklearn.metrics import roc_curve, auc

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label='{}, AUC value : {:.3}'.format(model_name, auc(fpr, tpr)))

plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()


# k-fold 교차 검증
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

create_model = create_custom_model(n_features, n_classes, 10, 3)
estimator = KerasClassifier(build_fn = create_model, epochs = 50, batch_size = 10, verbose = 2)
scores = cross_val_score(estimator, x_scale, y, cv = 10)
print('accuracy : {:0.2f}(+/-{:0.2f})'.format(scores.mean(), scores.std()))

# 모델3의 성능이 가장 우수
model = Sequential()
model.add(Dense(10, input_dim = 4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size = 10, verbose = 2)
print(model.evaluate(x_test, y_test))

y_pred = np.argmax(model.predict(x_test), axis = 1)
print('예측값 : ', y_pred)
real_y = np.argmax(y_test, axis = 1).reshape(-1, 1)
print('실제값 : ', real_y.ravel())
print('분류 실패 수 : ', (y_pred != real_y.ravel()).sum())

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(real_y, y_pred))
print(accuracy_score(real_y, y_pred))
print(classification_report(real_y, y_pred))

# 새로운 값으로 예측
new_x = [[5.5, 3.3, 1.2, 1.3], [3.5, 3.3, 0.2, 0.3], [1.5, 1.3, 6.2, 6.3]]
new_x = StandardScaler().fit_transform(new_x)
new_pred = model.predict(new_x)
print('예측값 : ', np.argmax(new_pred, axis = 1).reshape(-1, 1).flatten())