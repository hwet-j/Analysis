# XML로 제공된 강남구 도서관 정보 읽기
import urllib.request as req
import json

url = "http://openapi.seoul.go.kr:8088/sample/json/SeoulLibraryTime/1/5/"
plainText = req.urlopen(url).read().decode()
print(plainText)

jsonData = json.loads(plainText)
print(jsonData["SeoulLibraryTime"]["row"][0]["LBRRY_NAME"]) # [0]은 여러개 있는 row중에 맨앞에거를 선택

# get 함수 ---------
libData = jsonData.get("SeoulLibraryTime").get("row")
print(libData)

name = libData[0].get('LBRRY_NAME')
print(name)
print()
for msg in libData:
    name = msg.get("LBRRY_NAME")
    tel = msg.get("TEL_NO")
    addr = msg.get("ADRES")
    print(name + '\t' + tel + '\t' + addr)




