import pandas as pd
# pip install WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.font_manager as fm
 
fm.get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
plt.rc('font', family=font_name)

##############################################################################
# 백신 관련 파일 읽기
pnn_data = pd.read_excel('./data/result(AZ).xlsx')
# pnn_data = pd.read_excel('./data/result(PZ).xlsx')
pnn_data = pnn_data.drop('Unnamed: 0', axis=1)
print(pnn_data)
print(pnn_data.columns)

##############################################################################
# 각 백신별 count 그래프
ax = sns.countplot(pnn_data['label'], palette=sns.color_palette("Set1", 10))
ax.set_xticklabels(['negative','neutral','positive'])
plt.title('부정, 중립, 긍정 COUNT')
plt.xlabel('')
plt.legend()
plt.show()


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


##############################################################################
# 트위터 모양 클라우딩
twit_coloring = np.array(Image.open('./data/twit.png'))
from wordcloud import ImageColorGenerator
image_colors = ImageColorGenerator(twit_coloring)


# 부정단어 클라우딩
negative_words = ' '.join([word for word in negative_data['title']])
negative_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Reds').generate(negative_words)

fig, ax = plt.subplots(figsize=(12,6))
plt.imshow(negative_wc, interpolation='bilinear')
plt.axis('off')
plt.title('negative_wordcloud')
plt.savefig('./data/negative_wordcloud.png')
plt.show()


# 중립단어 클라우딩
neutral_words = ' '.join([word for word in neutral_data['title']])
print(neutral_words)
neutral_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Blues').generate(neutral_words)
  
fig, ax = plt.subplots(figsize=(12,6))
plt.imshow(neutral_wc, interpolation='bilinear')
plt.axis('off')
plt.title('neutral_wordcloud')
plt.savefig('./data/neutral_wordcloud.png')
plt.show()



#긍정 단어 클라우딩
positive_words = ' '.join([word for word in positive_data['title']])
positive_wc = WordCloud(font_path = font_location, background_color='white',width=1000, height=500, random_state=20, max_font_size = 120, mask = twit_coloring, colormap = 'Greens').generate(positive_words)
  
fig, ax = plt.subplots(figsize=(12,6))
plt.imshow(positive_wc, interpolation='bilinear')
plt.axis('off')
plt.title('positive_wordcloud')
plt.savefig('./data/positive_wordcloud.png')
plt.show()
