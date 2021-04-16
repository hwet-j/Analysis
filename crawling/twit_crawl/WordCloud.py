import pandas as pd
# pip install WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from konlpy.tag import Okt
from collections import Counter

import matplotlib.font_manager as fm
 
fm.get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)

##############################################################################
# 백신 관련 파일 읽기

def make_wc(search_text):
    data = pd.read_excel("./data/result(" +search_text +").xlsx")
    data = data.drop('Unnamed: 0', axis=1)
    print(data)
    print(data.columns)
    
    ##############################################################################
    # 각 백신별 count 그래프
    ax = sns.countplot(data['label'], palette=sns.color_palette("Set1", 10))
    ax.set_xticklabels(['negative','neutral','positive'])
    plt.title('(' + search_text + ')부정, 중립, 긍정 COUNT')
    plt.xlabel('')
    plt.legend()
    plt.savefig('Count_Sentiment(' + search_text + ').png')
    plt.show()
    
    
    ###############################################################################
    
    # 한글빼고 제거
    data['title'] = data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    
    data['title'] = data['title'].str.replace('^ +', "") # white space 데이터를 empty value로 변경(긴 공백이나 특수문자들로만 이루어진데이터를 공백하나로 바꿔줌)
    data['title'].replace('', np.nan, inplace=True)   # 공백으로 바꿔준 데이터를 Null값으로 
    # print(data.isnull().sum())   # Null값 존재확인 
    data = data.dropna(how = 'any') # Null값 제거
    # print(len(data)) 
    
    # 불용어 정의
    stopwords_read = pd.read_csv("./dictionary/stopwords_self.txt", sep="\n", header=None)
    # print(stopwords_read.columns)
    stopwords = list(stopwords_read[0])
    # print(stopwords, type(stopwords))
    # 형태소 분석
    okt = Okt()
    # 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장
    datas = []
    for sentence in data['title']:
        temp = []
        temp = okt.nouns(sentence) # 토큰화
        temp = [word for word in temp if not word in stopwords] # 불용어 제거
        temp = [word for word in temp if len(word) > 1] # 1글자 제거
        for word in temp:
            datas.append(word)
    count = Counter(datas)
    
    ###############################################################################
    
    '''
    ##############################################################################
    # 부정 단어
    negative_data = pnn_data['label'] == -1
    negative_data = pnn_data[negative_data]
    negative_data = negative_data.drop('label', axis=1)
    print(negative_data)
    # 중립 단어
    neutral_data = pnn_data['label'] == 0
    neutral_data = pnn_data[neutral_data]
    neutral_data = neutral_data.drop('label', axis=1)
    print(neutral_data)
    # 긍정 단어
    positive_data = pnn_data['label'] == 1
    positive_data = pnn_data[positive_data]
    positive_data = positive_data.drop('label', axis=1)
    print(positive_data)
    '''
    
    ##############################################################################
    # 트위터 모양 (빈도수기준) 클라우딩
    twit_coloring = np.array(Image.open('twit.png'))
    if search_text == '백신':
        twit_coloring = np.array(Image.open('v_image.png'))
    from wordcloud import ImageColorGenerator
    image_colors = ImageColorGenerator(twit_coloring)
    
    covid_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring).generate_from_frequencies(count)
   
    fig, ax = plt.subplots(figsize=(12,6))
    plt.imshow(covid_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('(' + search_text + ')wordcloud')
    plt.savefig('wordcloud(' + search_text + ').png')
    plt.show()
    
    '''
    # (문장 자체에서 단어를 뽑아 클라우딩)
    # 부정단어 클라우딩
    negative_words = ' '.join([word for word in negative_data['title']])
    negative_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Reds').generate(negative_words)
    
    fig, ax = plt.subplots(figsize=(12,6))
    plt.imshow(negative_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('(' + search_text +')negative_wordcloud')
    plt.savefig('negative_wordcloud(' + search_text + ').png')
    plt.show()
    
    
    # 중립단어 클라우딩
    neutral_words = ' '.join([word for word in neutral_data['title']])
    neutral_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Blues').generate(neutral_words)
      
    fig, ax = plt.subplots(figsize=(12,6))
    plt.imshow(neutral_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('(' + search_text +')neutral_wordcloud')
    plt.savefig('neutral_wordcloud(' + search_text + ').png')
    plt.show()
    
    
    
    #긍정 단어 클라우딩
    positive_words = ' '.join([word for word in positive_data['title']])
    positive_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Greens').generate(positive_words)
      
    fig, ax = plt.subplots(figsize=(12,6))
    plt.imshow(positive_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('(' + search_text +')positive_wordcloud')
    plt.savefig('positive_wordcloud(' + search_text + ').png')
    plt.show()
    '''
