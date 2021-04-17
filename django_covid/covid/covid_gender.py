import requests
import xmltodict
import json
from pprint import pprint



url_vaccin = "https://nip.kdca.go.kr/irgd/cov19stats.do?list=all"




    
res = requests.get(url_vaccin)
# print(res.url)
print(res.text)
    
    
# xml -> dict
dict_data = xmltodict.parse(res.text)
# print(dict_data)
    
# dict -> json
json_data = json.dumps(dict_data)
# print(json_data,type(json_data))
    
# json -> dict
dict_data = json.loads(json_data)
print(dict_data, type(dict_data))

# total Cnt Check
# totalCount = dict_data['response']['body']['totalCount']
# if totalCount == "0":
    # return False
    
# 지역정보를 담은 리스트 저장
vaccin_data = dict_data['response']['body']['items']['item']
vaccin_data.reverse()
# pprint(gender_data)
for a in vaccin_data:
    print(a)
# return vaccin_data
print(vaccin_data[0])
print(vaccin_data[2])