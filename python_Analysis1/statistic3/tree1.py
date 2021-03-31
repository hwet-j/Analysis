# 의사결정나무(Decision Tree) - CART - Classification과 Regression 모두 가능
# 여러 규칙을 순차적으로 적용하면서 분류나 에측을 진행하는 단순 알고리즘 사용 모델

import pydotplus
from sklearn import tree
import collections

# height, hair로 남녀 구분
x = [[180, 15],
     [177, 45],
     [156, 35],
     [174, 5],
     [166, 33],
]

y = ['man','woman','woman','man','woman']
label_names = ['height', 'hair length']

model = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
print(model)
fit = model.fit(x, y)
print('acc : {:.3f}'.format(model.score(x, y)) )

mydata = [[178, 8]]
pred = model.predict(mydata)
print('pred : ',pred)

# 시각화 - graphviz 툴을 사용
dot_data = tree.export_graphviz(model, feature_names = label_names, out_file = None,\
                                filled = True, rounded = True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'blue')
edges = collections.defaultdict(list) # list type변수

for e in graph.get_edge_list():
    edges[e.get_source()].append(int(e.get_destination()))
    
for e in edges:
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph.write_png('tree.png') # 이미지 저장   

import matplotlib.pyplot as plt
img = plt.imread('tree.png')
plt.imshow(img)
plt.show()

