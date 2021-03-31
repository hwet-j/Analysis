# BeautifulSoup의 find(), select() 

from bs4 import BeautifulSoup

html_page = """
<html><body>
<h1>제목 태그</h1>
<p>웹 문서 읽기</p>
<p>원하는 자료 선택</p>
<h1>끝</h1>
</body></html>
"""

#print(type(html_page))    # <class 'str'>
#print(html_page)    # html형태를 가진 문자열    

soup = BeautifulSoup(html_page, 'html.parser')  # BeautifulSoup 객체생성
#print(type(soup))   #<class 'bs4.BeautifulSoup'> BeautifulSoup이 제공하는 명령 사용가능

h1 = soup.html.body.h1 # soup에 html안에 body안에 h1태그(태그까지)
print("h1 : ", h1.string)   # 뽑아온 태그의 내용

p1 = soup.html.body.p   # 최초의 p를 만남
print("p1 : ", p1.string)

p2 = p1.next_sibling.next_sibling
print("p2 : ", p2.string)


print("\n\nfind()를 사용")
html_page2 = """
<html><body>
<h1 id="title">제목 태그</h1>
<p>웹 문서 읽기</p>
<p id="my">원하는 자료 선택</p>
<h1>끝</h1>
</body></html>
"""

soup2 = BeautifulSoup(html_page2, 'html.parser')
print(soup2.p, ' ', soup2.p.string) # 직접 최초 tag선택
print(soup2.find('p').string)
print(soup2.find('p', id='my').string)  # p태그의 id값을 선택해줘 출력
print(soup2.find_all('p'))  # p태그를 전부 불러옴(태그도 같이 가져옴)
print(soup2.find(id='title').string)    # id값만 선택하여 출력
print(soup2.find(id='my').string)

print("\n\nfind_all(), findAll()를 사용")
html_page3 = """
<html><body>
<h1 id="title">제목 태그</h1>
<p>웹 문서 읽기</p>
<p id="my">원하는 자료 선택</p>
<div>
    <a href="https://www.naver.com">naver</a><br>
    <a href="https://www.daum.net">daum</a><br>
</div>
</body></html>
"""

soup3 = BeautifulSoup(html_page3, 'html.parser')
print(soup3.find('a'))
print(soup3.find('a').string)
print(soup3.find(['a']))
print(soup3.find_all(['a']))
print(soup3.findAll(['a']))
print(soup3.find_all('a'))
print(soup3.find_all(['a', 'p']))   # a태그, p태그 둘다 가져옴
#print(soup3)
#print(soup3.prettify())

print()
links = soup3.find_all('a')
print(links)
for i in links:
    href = i.attrs['href']
    text = i.string
    print(href, ' ', text)
    
print('\n\nfind() 정규 표현식 사용')
import re
# https로 시작되는 데이터 
link2 = soup3.find_all(href=re.compile(r'^https'))
print(link2)
for k in link2:
    print(k.attrs['href'])

print("\n\nselect()사용 (css의 selector)")
html_page4 = """
<html><body>
<div id="hello">
    <a href="https://www.naver.com">naver</a><br>
    <span>
        <a href="https://www.daum.net">daum</a>
    </span><br>
    
    <ul class="world">
        <li>안녕</li>
        <li>반가워</li>
    </ul>
</div>
<div id="hi">
second div
</div>

</body></html>
"""
soup4 = BeautifulSoup(html_page4, 'lxml')
aa = soup4.select_one("div#hello a").string    # 모든 "div#hello a" a태그를 읽어와야 하지만 select_one이라 하나만 읽음 
#aa = soup4.select_one("div#hello > a").string  # 직계 바로 아래 자식에 존재하는 a태그만
print('aa : ', aa)

bb = soup4.select("div#hello ul.world > li")    # 복수 선택     # div#hello ul : 자손 div#hello > ul : 직계
print('bb : ', bb)  # bb :  [<li>안녕</li>, <li>반가워</li>]
for i in bb:
    print('li : ', i.string)

