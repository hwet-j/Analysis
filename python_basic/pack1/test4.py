# 정규 표현식
# 특정한 규칙을 가진 문자열의 집합을 표현하는데 사용하는 형식 언어. 
# https://velog.io/@ednadev/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D%EA%B3%BC-re%EB%AA%A8%EB%93%88
import re
from re import IGNORECASE

ss = '1234 abc가나다ABC_555_6_7989_555kbs가나 다라 가나 마 정규 표현식 특정한 규칙'
print(ss)
# findall ''안에 여러개 조건 대입가능
print(re.findall(r'125', ss))   # 찾는 데이터 없으면 빈 list 객체
print(re.findall(r'123', ss))   # 하나면 한개만
print(re.findall(r'가나', ss))    # 여러개면 존재하는 수만큼

# [] 문자들의 범위를 나타내기 위해 사용
# [abck] : a or b or c or k
# [abc.^] : a or b or c or . or ^
# [a-d] : -와 함께 사용되면 해당 문자 사이의 범위에 속하는 문자 중 하나
# [0-9] : 모든 숫자
# [a-z] : 모든 소문자
# [A-Z] : 모든 대문자
# [a-zA-Z0-9] : 모든 알파벳 문자 및 숫자
# [^0-9] : ^가 맨 앞에 사용 되는 경우 해당 문자 패턴이 아닌 것과 매칭    
print(re.findall(r'[0,1,3]', ss))   # 0,1,3을 다 찾음 
print(re.findall(r'[0,1,5]', ss))   
print(re.findall(r'[0-9]', ss))     # 0~9까지 다 찾음

print('\n<<< +, ?, {n}, {a,b} >>>')
print(re.findall(r'[0-8]+', ss))    # 뒤에+ 0~8의 데이터를 이어서져있으면 이어서 저장
print(re.findall(r'[0-9]?', ss))    # 뒤에? 0~9까지의 데이터면 그 데이터를 잘라서 저장 아니면 빈value로 저장됨
print(re.findall(r'[a-z,A-Z]+', ss))    # 영어데이터를 뽑아옴
print(re.findall(r'[가-힣]+', ss))    # 한글 데이터를 뽑아옴
print(re.findall(r'[가-힣]{2}', ss))  # 한글 데이터를 2글자이상의 데이터에서 2글자만 뽑아옴
print(re.findall(r'[가-힣]{2,3}', ss))    # 한글 데이터를 2글자 이상의 데이터에서 2,3글자를 뽑아옴

print('\n^')
# ^ : 문자열의 맨 앞부터 일치하는 경우 검색
# $ : 문자열의 맨 뒤부터 일치하는 경우 검색
# ^ = 시작, $ = 끝 : 각각 문자열의 시작과 끝을 의미
print(re.findall(r'[^0-9]+', ss))
print(re.findall(r'^1', ss))
print(re.findall(r'^2', ss))
print(re.findall(r'칙$', ss))

# \ : 다른 문자와 함께 사용되어 특수한 의미로 사용
# \d : 숫자를 [0-9]와 동일
# \D : 숫자가 아닌 문자 [^0-9]와 동일
# \s : 공백 문자(띄어쓰기, 탭, 엔터 등)
# \S : 공백이 아닌 문자
# \w : 알파벳대소문자, 숫자 [0-9a-zA-Z]와 동일
# \W : non alpha-numeric 문자 [^0-9a-zA-Z]와 동일
# \t, \n, \r - tab, newline, return
print(re.findall(r'\d', ss))
print(re.findall(r'\d{2}', ss))
print(re.findall(r'\d{2,3}', ss))
