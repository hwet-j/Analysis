# 사용자 정의 모듈
a = 10
print(a)

def aa():
    print('aa 만세')
    
# 외부 모듈의 멤버 사용하기
import pack2.test12my # 경로 지정

print('tot :', pack2.test12my.tot)

li1 = [1, 2]
li2 = [3, 4]
pack2.test12my.ListHap(li1, li2)

def abc():
        if __name__ == '__main__':
            print('응용 프로그램이 시작되는 모듈')
abc()

print()
# 외부 모듈의 멤버 사용하기 2
from pack2 import test12my
test12my.mbc()

from pack2.test12my import mbc
mbc()

from pack2.test12my import mbc, kbs, tot
mbc()
kbs()
print(tot)

# 외부 모듈의 멤버 사용하기 3
from other.test12our import Hap, Cha
print(Hap(5, 3))
print(Cha(5, 3))

# anaconda/Lib 에 만들어 놓은 모듈을 넣어두면 패키지 선언필요없이 모듈명만 선언하여 호출 가능
# PyDev -> interpreters -> Python interpreter -> Libraries 에서 모듈을 읽어주는 경로 확인 및 지정 가능
import test12our2
print(test12our2.Gop(5, 3))
print(test12our2.Nanugi(5, 3))

import math
print(math.pi)

from test12our2 import Gop
print(Gop(4, 3))

from math import pi
print(pi)





