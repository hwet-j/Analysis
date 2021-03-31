# Word를 수치화해서 vector에 담기
import numpy as np

# 단어 one-hot encoding
data_list = ['python','lan','program','computer','say']
print(data_list)

values = []
for x in range(len(data_list)):
    values.append(x)
    
print(values)

values_len = len(values)
print(values_len)

one_hot = np.eye(values_len)
print(one_hot)

print('---------------')
from gensim.models import word2vec
# word2vec
sentence = [['python','lan','program','computer','say']]
model = word2vec.Word2Vec(sentences=sentence, min_count=1,size=50)
print(model.wv)
word_vectors = model.wv
print('word_vectors.vocab : ', word_vectors.vocab)  # key,value로 구성된 vocab obj
print()
vocabs = word_vectors.vocab.keys()
print('vocabs : ', vocabs)
vocabs_val = word_vectors.vocab.values()
print('vocabs_val : ', vocabs_val)

print()
word_vectors_list = [word_vectors[v] for v in vocabs]
print(word_vectors_list)

print('-----------------')
print(word_vectors.similarity(w1='lan',w2='program'))   # 단어 간 유사도 거리
print(word_vectors.similarity(w1='lan',w2='say'))
print()
print(model.wv.most_similar(positive='lan'))
# [('computer', 0.2277611345052719), ('program', 0.035543810576200485), ('say', -0.10670819878578186), ('python', -0.15748611092567444)]








