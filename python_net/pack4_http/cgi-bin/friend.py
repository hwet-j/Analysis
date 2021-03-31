import cgi

form = cgi.FieldStorage()
name = form['name'].value
phone = form['phone'].value
gender = form['gen'].value

# 이 값으로 DB에 저장, 연산등에 작업을 할 수 있다.
# 여기서는 단순히 출력에 참여

print('Content-Type:text/html;charset=utf-8\n')

print("""
<html>
<body>
이름은 {}, 전화는 {}, 성별은 {}
</body>
</html>
""".format(name, phone, gender))