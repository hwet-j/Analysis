# XML로 제공된 강남구 도서관 정보 읽기
import urllib.request as req
from bs4 import BeautifulSoup

url = "http://openapi.seoul.go.kr:8088/sample/xml/SeoulLibraryTime/1/5/"
plainText = req.urlopen(url).read().decode()
#print(plainText)

xmlObj = BeautifulSoup(plainText, 'lxml')
libData = xmlObj.select("row")
#print(libData)

for data in libData:
    name = data.find("lbrry_name").text
    addr = data.find("adres").text
    print('도서관명 : ',name)
    print('주 소 : ',addr)


