# 검색 결과를 형태소 분석하여 단어 빈도수를 구하고 이를 기초로 위드클리우드 차트 출력
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

#keyword = input('검색어:')
keyword = "백신"
print(quote(keyword))

# 동아일보 검색 기능
target_url = "https://www.donga.com/news/search?query=" + quote(keyword)
print(target_url)
source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf8')
#print(soup)

msg = ""

for title in soup.find_all("p", "tit"):
    title_link = title.select('a')
    #print(title_link)
    article_url = title_link[0]['href']
    #print(article_url)
    
    try:
        source_article = urllib.request.urlopen(article_url)    # 해당 사이트에 들어가 실제 기사 읽기
        soup = BeautifulSoup(source_article,'lxml', from_encoding='utf-8')
        contents = soup.select('div.article_txt')
        #print(contents)
        
        for imsi in contents:
            item = str(imsi.find_all(text=True))
            #print(item)
            msg = msg + item
            
    except Exception as e:
        pass

print(msg)

from konlpy.tag import Okt
from collections import Counter
# 형태소 분석을 해서 2글자이상의 명사만 뽑아내는 작업
nlp = Okt()
nouns = nlp.nouns(msg)
result = []
for i in nouns:
    if len(i) > 1:  # 2글자 이상을 취하게함
        result.append(i)
        
print(result)
print(len(result))
count = Counter(result) # 단어별 개수 확인
print(count)    
tag = count.most_common(50) # 상위 50개만 작업에 참여

print()
# 이런 작업을 처음한다면 anaconda prompt창에서 install해줘야함
# pip install simplejson
# pip install pytagcloud
import pytagcloud

taglist = pytagcloud.make_tags(tag, maxsize=100)
print(taglist)

pytagcloud.create_tag_image(taglist, 'word.png', size=(1000, 600), fontname='Korean', rectangular=False)

# 저장된 이미지 읽기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inlie

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()

# 브라우저로 출력
import webbrowser
webbrowser.open("word.png")

