# 웹 뉴스 정보를 읽어 형태소 분석 -> 단어별 유사도 출력

import pandas as pd
from konlpy.tag import Okt

okt = Okt()

with open('news.txt', mode='r', encoding='utf8') as f:
    #print(f.read())
    lines = f.read().split('\n')
    #print(len(lines))
    
wordDic = {}    # 단어 수 확인을 위한 dict type

for line in lines:
    datas = okt.pos(line)   # 품사 태깅
    #print(datas)
    
    for word in datas:
        if word[1] == 'Noun':   # 명사만 작업에 참여
            #print(word)
            #print(word[0] in wordDic)
            if not(word[0] in wordDic):
                wordDic[word[0]] = 0
            wordDic[word[0]] += 1
            
print(wordDic)

# 단어 건수별 내림차순 정렬
keys = sorted(wordDic.items(), key=lambda x:x[1], reverse=True)
print(keys)

# DataFrame에 담기 - 단어와 건수
wordList = []
countList = []

for word, count in keys[:20]: # 상위 20개만 작업참여
    wordList.append(word)
    countList.append(count)
    
df = pd.DataFrame()
df['word'] = wordList
df['count'] = countList
print(df)
#-----------------------

print()
# word2vec
results = []

with open('news.txt', mode='r', encoding='utf8') as fr:
    lines = fr.read().split('\n')
    for line in lines:
        datas = okt.pos(line, stem=True)    # 원형 어근 형태로 처리 
        print(datas)
        imsi = []
        for word in datas:
            if not word[1] in ['Punctuation','Suffix','Josa','Verb','Modifier','Number','Determiner','Foreign']:
                imsi.append(word[0])
        imsi2 = (" ".join(imsi)).strip()
        results.append(imsi2)
        
print(results)                

fileName = 'news2.txt'            
with open(fileName, mode='w', encoding='utf8') as fw:
    fw.write('\n'.join(results))
    print('저장 성공')
            
print('--------------------')
# Word Embedding (단어를 수치화)의 일종으로 word2vec
from gensim.models import word2vec
genObj = word2vec.LineSentence(fileName)
print(genObj)   #<gensim.models.word2vec.LineSentence object at 0x00000163F4F16730>

# 모델 생성

model = word2vec.Word2Vec(genObj, size = 100, window=10, min_count=2, sg=1) # sg=0 CBOW, sg=1 Skip-Gram
print(model)
#model.init_sims(replace=True)   # 필요없는 메모리 해제

# 학습시킨 모델은 저장후 재사용 가능
try:
    model.save('news.model')
except Exception as e:
    print('err : ', e)

model = word2vec.Word2Vec.load('news.model')
print(model)
print(model.wv.most_similar(positive=['질병']))
print(model.wv.most_similar(positive=['질병'], topn=3))
print(model.wv.most_similar(positive=['질병','예방접종'], topn=3))
# positive : 단어 사전에 해당 단어가 있을 확률
# negative : 단어 사전에 해당 단어가 없을 확률



