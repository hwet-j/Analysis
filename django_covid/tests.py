from django.test import TestCase
from datetime import date, timedelta
from covid import covid
# Create your tests here.

now = date.today()
now_str = now.strftime("%Y%m%d")
print(now.strftime("%Y%m%d"))

data = covid.get_corona_data(now_str,now_str)
if not data : 
    yesterday = now - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d")
    print(yesterday_str)
    
    data = covid.get_corona_data(yesterday_str,yesterday_str)

print(data)