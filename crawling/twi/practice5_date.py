from datetime import datetime, timedelta

print(datetime.today().strftime("%Y-%m-%d"))
today = datetime.today().strftime("%Y-%m-%d")
yesterday = datetime.today() - timedelta(2) # timedelta의 숫자를 하나씩 늘려주면서 ..
print(yesterday.strftime("%Y-%m-%d"))

for i in range(5):
    print((datetime.today() - timedelta(i)).strftime("%Y-%m-%d"))
