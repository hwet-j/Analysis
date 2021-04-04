# Module : 소스 코드의 재사용을 가능하게 하며, 소스 코드를 하나의 이름 공간으로 구분하고 관리할 수 있다. 
# 멤버 : 일반 명령문 , 함수, 모듈, 클래스
# 하나의 파일로 처리된다.
# 내장된 표준 모듈, 사용자 정의 모듈, 제 3자 모듈(third party)

# 내장된 표준 모듈(로딩 필요없음) 일부 사용해보기
print('뭔가를 하다가.. 외부 모듈 필요하면 import module명 하고 사용')
print(sum([2,3]))

import sys
print('모듈 경로 :', sys.path)
# sys.exit() # 프로그램 강제종료
print('프로그램 끝')

import math
print(math.pi)
print(math.sin(math.radians(30)))

import calendar
calendar.setfirstweekday(6)
calendar.prmonth(2021,2)
print()

# 난수 출력
import random
print(random.random())
print(random.randint(1,10))

from random import random
print(random())
from random import randint
print(randint(1,10))
