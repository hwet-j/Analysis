# 예외처리 : try ~ except - 예상치 않은 에러에 대응하기 위한 문법

def divide(a, b):
    return a / b

print('이런 저런 작업을 하다가...')

c = divide(5, 2)
# c = divide(5, 0)    # ZeroDivisionError: division by zero
print(c)
print('----------')

try:
    c = divide(5, 2)
    # c = divide(5, 0)
    print(c) 
    
    aa = [1,2] 
    print(aa[0])
    # print(aa[2])
    
    f = open('c:/abc.txt')
    
except ZeroDivisionError:
    print('두번째 숫자는 0을 주지 마세요')
except IndexError as e:
    print('인덱스 에러 : ', e)
except Exception as err:
    print('에러 발생 : ', err)
finally:
    print('에러 유무 관계없이 반드시 수행')

print('프로그램 끝')