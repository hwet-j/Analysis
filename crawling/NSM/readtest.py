import pandas as pd
from datetime import datetime, timedelta
# with open("test.txt", encoding='utf-8') as text: 
#     data = text.readlines() 

# list = []
# for i in data:  # \n가 출력되어 이를 제거함
#     data_list = i.replace("\n", "")
#     list.append(data_list)
#     
# print(list)

with open("datas.csv", encoding='utf-8') as text: 
    data = text.readlines() 

print(data)

