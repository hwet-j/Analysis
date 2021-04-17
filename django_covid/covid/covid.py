import requests
import xmltodict
import json
from pprint import pprint

url = "http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19SidoInfStateJson"
url_gender = "http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19GenAgeCaseInfJson"
serviceKey_1 = '2PLDIVNdhJ3pWgmyk2qXL2LekLrwfv5r8z2tq6aJcJHC6nHNToX8ui60+hPuZDOofzC8C6xXpKFAQizM9nqtEA=='
serviceKey_2 = 'h92s7QADkL1WM1LYJeYFzOx4Fmh+CV5s/Vhl0jrrZEYB3WV3ipW2sE1Had2/upfDhMMbLr1M0V6/DTL6YaiLKw=='
serviceKey_gender1= '2PLDIVNdhJ3pWgmyk2qXL2LekLrwfv5r8z2tq6aJcJHC6nHNToX8ui60%2BhPuZDOofzC8C6xXpKFAQizM9nqtEA%3D%3D'
serviceKey_gender2='2PLDIVNdhJ3pWgmyk2qXL2LekLrwfv5r8z2tq6aJcJHC6nHNToX8ui60+hPuZDOofzC8C6xXpKFAQizM9nqtEA=='
url_vaccin = "https://nip.kdca.go.kr/irgd/cov19stats.do?list=all"

# 백신 접종 정보 얻기.
def get_vaccin_data():
    res = requests.get(url_vaccin)

    # xml -> dict
    dict_data = xmltodict.parse(res.text)
    # print(dict_data)
        
    # dict -> json
    json_data = json.dumps(dict_data)
    # print(json_data,type(json_data))
        
    # json -> dict
    dict_data = json.loads(json_data)
    print(dict_data, type(dict_data))
    
    # 지역정보를 담은 리스트 저장
    vaccin_data = dict_data['response']['body']['items']['item']
    vaccin_data.reverse()
    # pprint(gender_data)
    for a in vaccin_data:
        print(a)
    return vaccin_data;  
def get_vaccin_time():
    res = requests.get(url_vaccin)

    # xml -> dict
    dict_data = xmltodict.parse(res.text)
    # print(dict_data)
        
    # dict -> json
    json_data = json.dumps(dict_data)
    # print(json_data,type(json_data))
        
    # json -> dict
    dict_data = json.loads(json_data)
    #print(dict_data, type(dict_data))
    
    # 지역정보를 담은 리스트 저장
    vaccin_time = dict_data['response']['body']['dataTime']
    return vaccin_time;  

#연령 세대별 확진자 정보 얻기.
def get_gender_data(startCreateDt, endCreateDt):
    params = {
        'serviceKey' : serviceKey_gender2,
        'pageNo' : '1',
        'numOfRows' : 10,
        'startCreateDt': startCreateDt,
        'endCreateDt': endCreateDt,
    }
    res = requests.get(url=url_gender , params=params)
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
    #   return False
        
    # 연령정보를 담은 리스트 저장
    gender_data = dict_data['response']['body']['items']['item']
    gender_data.reverse()
    # pprint(gender_data)
    for a in gender_data:
        print(a)
    return gender_data
    
    
#코로나 시도별 확진자 정보 얻기
def get_corona_data(startCreateDt, endCreateDt):
    params = {
        'serviceKey' : serviceKey_1, 
        'pageNo' : '1',
        'numOfRows' : 10,
        'startCreateDt': startCreateDt,
        'endCreateDt': endCreateDt,
    }
    
    res = requests.get(url=url , params=params)
    # print(res.url)
    # print(res.text)
    
    # xml -> dict
    dict_data = xmltodict.parse(res.text)
    # print(dict_data)
    
    # dict -> json
    json_data = json.dumps(dict_data)
    # print(json_data,type(json_data))
    
    # json -> dict
    dict_data = json.loads(json_data)
    # print(dict_data, type(dict_data))
    # pprint(dict_data['response']['body']['items']['item'])
    
    # total Cnt Check
    totalCount = dict_data['response']['body']['totalCount']
    if totalCount == "0":
        return False
    
    # 지역정보를 담은 리스트 저장
    area_data = dict_data['response']['body']['items']['item']
    area_data.reverse()
    # pprint(area_data)
    for a in area_data:
        print(a)
    return area_data