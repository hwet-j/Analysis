# SVM 모델로 이미지 분류 
from sklearn.datasets import fetch_lfw_people   #  공인 얼굴 지원
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


faces = fetch_lfw_people(min_faces_per_person = 60)
# print(faces)
# print(faces.DESCR)

print(faces.data)
print(faces.data.shape) # (1348, 2914)
print(faces.target)
print(faces.target_names)
print(faces.images.shape)   # (1348, 62, 47)


# print(faces.images[1])
# print(faces.target_names[faces.target[1]])
# plt.imshow(faces.images[1], cmap='bone')
# plt.show()

# 여러개
# fig, ax = plt.subplots(3, 5)    # 3행 5열
# print(fig)  # Figure(640x480)
# print(ax.flat)
# print(len(ax.flat))


# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap="bone")
#     axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
#  
# plt.show()


# 주성분 분석으로 이미지 차원을 축소시켜 분류작업을 진행
m_pca = PCA(n_components = 150, whiten = True, random_state = 0)
m_svc = SVC(C=1)
model = make_pipeline(m_pca, m_svc)
print(model)

# train / test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state = 1)
print(x_train[0], x_train.shape)    # (1011, 2914)
print(y_train[0], y_train.shape)    # (1011,)


model.fit(x_train, y_train)
pred = model.predict(x_test)
print('pred : ', pred[:10])
print('real : ', y_test[:10])

print()
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, target_names = faces.target_names))

#                    precision    recall  f1-score   support
# 
#      Ariel Sharon       1.00      0.50      0.67        14
#      Colin Powell       0.85      0.85      0.85        54
#   Donald Rumsfeld       1.00      0.50      0.67        30
#     George W Bush       0.69      0.99      0.81       134
# Gerhard Schroeder       0.96      0.71      0.81        31
#       Hugo Chavez       1.00      0.45      0.62        20
# Junichiro Koizumi       1.00      0.42      0.59        12
#        Tony Blair       0.97      0.76      0.85        42
# 
#          accuracy                           0.80       337
#         macro avg       0.93      0.65      0.73       337
#      weighted avg       0.85      0.80      0.79       337


from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(y_test, pred)
# print('confusion_matrix : ', confusion_matrix(y_test, pred))
print('confusion_matrix : ', mat)
print('acc : ', accuracy_score(y_test, pred))   # 0.79525222

# 분류결과를 시각화
# x_test[0]째 1개만 미리보기
# plt.subplots(1, 1)
# print(x_test[0], ' ', x_test[0].shape)  # (2914,)
# print(x_test[0].reshape(62, 47))    # 1차원을 2차원으로 변환해야 이미지 출력
# plt.imshow(x_test[0].reshape(62, 47), cmap='bone')
# plt.show()

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62, 47), cmap="bone")
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1], color = 'black' if pred[i] == y_test[i] else 'red')
    fig.suptitle('pred result',size = 14)
   
plt.show()


# 오차행렬 시각화
import seaborn as sns
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False, \
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true(real) label')
plt.ylabel('predicted label')
plt.show()








