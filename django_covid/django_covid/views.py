from django.shortcuts import render
import requests
import json

from covid import covid
from datetime import date, timedelta
from flask.templating import render_template

def list_gender(request):
    now = date.today()
    now_str = now.strftime("%Y%m%d")
       
    # 세대 성별 데이터 추출.
    print(now_str)
    gdata =  covid.get_gender_data(now_str,now_str)
    # gdata = [1,2,3,4,5,6,7,8,9,0,0,0,0,]
    # print(gdata)
    
    return render(request, 'list_gender.html',
                  { 'data_male': gdata[0], 'data_female':gdata[1],
                   'data_80':gdata[2],'data_70':gdata[3],'data_60':gdata[4],'data_50':gdata[5],'data_40':gdata[6],'data_30':gdata[7],'data_20':gdata[8],'data_10':gdata[9],'data_00':gdata[10],'data_g':gdata})
    
def list_vaccin(request):
    v_data = covid.get_vaccin_data()
    v_time = covid.get_vaccin_time()
    return render(request, 'list_vaccin.html',{'day0_v': v_data[2],'day_1':v_data[1],'day_sum':v_data[0],'v_time':v_time})    

def list(request):
    now = date.today()
    now_str = now.strftime("%Y%m%d")
       
    data =  covid.get_corona_data(now_str,now_str)
    print(data)
    
    if not data :
        print("not today data")
        now = now - timedelta(days=1)
        now_str = now.strftime("%Y%m%d")
        print(now_str)
        data = covid.get_corona_data(now_str,now_str)
        print(data)
    
    yesterday = now - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")
    weekagoday = now - timedelta(days=7)
    weekagoday_str = weekagoday.strftime("%Y%m%d")
    monthagoday = now - timedelta(days=30)
    monthagoday_str = monthagoday.strftime("%Y%m%d")    
    print(now_str,"/",yesterday_str,"/",weekagoday_str,"/",monthagoday_str)
    
    now_1 = now - timedelta(days=1)
    now_1_str = now_1.strftime("%Y%m%d")
    now_2 = now - timedelta(days=2)
    now_2_str = now_2.strftime("%Y%m%d")
    now_3 = now - timedelta(days=3)
    now_3_str = now_3.strftime("%Y%m%d")
    now_4 = now - timedelta(days=4)
    now_4_str = now_4.strftime("%Y%m%d")
    now_5 = now - timedelta(days=5)
    now_5_str = now_5.strftime("%Y%m%d")
    now_6 = now - timedelta(days=6)
    now_6_str = now_6.strftime("%Y%m%d")
    now_7 = now - timedelta(days=7)
    now_7_str = now_7.strftime("%Y%m%d")
    
    data_2 = covid.get_corona_data(now_2_str,now_2_str)
    # print(data_2)
    data_3 = covid.get_corona_data(now_3_str,now_3_str)
    # print(data_3)
    data_4 = covid.get_corona_data(now_4_str,now_4_str)
    # print(data_4)
    data_5 = covid.get_corona_data(now_5_str,now_5_str)
    # print(data_5)
    data_6 = covid.get_corona_data(now_6_str,now_6_str)
    # print(data_6)
    
    dataYesterday = covid.get_corona_data(yesterday_str,yesterday_str)
    dataWeek = covid.get_corona_data(weekagoday_str,weekagoday_str)
    dataMonth = covid.get_corona_data(monthagoday_str,monthagoday_str)
    print(type(data))
    print(data[0])
    
    # 세대 성별 데이터 추출.
    print(now_str)
    gdata =  covid.get_gender_data(now_str,now_str)
    # gdata = [1,2,3,4,5,6,7,8,9,0,0,0,0,]
    # print(gdata)
    
    # 세대 성별 데이터 추출.
    print(now_str)
    gdata =  covid.get_gender_data(now_str,now_str)
    # gdata = [1,2,3,4,5,6,7,8,9,0,0,0,0,]
    # print(gdata)
    
    # return render_template('list.html',data = data)
    '''
    return render(request, 'list.html',
                  {'data' : data,'data_2' : data_2[0],'data_3' : data_3[0],'data_4' : data_4[0],'data_5' : data_5[0],'data_6' : data_6[0], 'koreaData':data[0],'dataYesterday':dataYesterday[0],'dataWeek':dataWeek[0],'dataMonth':dataMonth[0],'today':now_str})
    '''
    return render(request, 'list.html',
                  {'data' : data,'data_2' : data_2[0],'data_3' : data_3[0],'data_4' : data_4[0],'data_5' : data_5[0],'data_6' : data_6[0], 'koreaData':data[0],'dataYesterday':dataYesterday[0],'dataWeek':dataWeek[0],'dataMonth':dataMonth[0],'today':now_str
                   ,'data_male': gdata[0], 'data_female':gdata[1],'data_80':gdata[2],'data_70':gdata[3],'data_60':gdata[4],'data_50':gdata[5],'data_40':gdata[6],'data_30':gdata[7],'data_20':gdata[8],'data_10':gdata[9],'data_00':gdata[10],'data_g':gdata})
    

def home(request):
    return render(request, 'home.html', {})
