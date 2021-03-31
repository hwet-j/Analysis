# 클라이언트 서버로 자료전달

import cgi

# 사용자(client)가 입력한 자료를 받기 - get
form = cgi.FieldStorage()   
irum = form['name'].value   # 자바의 request.getparameter("name")라고 생각하면됨
nai = form['age'].value

print('Content-Type:text/html;charset=utf-8\n')

print("""
<html>
<body>
사용자가 입력한 자료는 
이름은 {0}
나이는 {1}

</body>
</html>
""".format(irum, nai))