# GIL이란 Global Interpreter Lock의 약자로 파이썬 인터프리터가 한 스레드만 하나의 
# 바이트코드를 실행 시킬 수 있도록 해주는 Lock입니다.
# 하나의 스레드에 모든 자원을 허락하고 그 후에는 Lock을 걸어 다른 스레드는 실행할 수 없게 막아버리는 것
# 이런 이유로 스레드는 구조적으로 충돌이 발생하는 경우가 있다. 
# 그래서 멀티 프로세싱 모듈을 지원한다.

# Pool : 입력 값에 대해 process들을 건너건너 분배하여 함수실행을 병렬화하는 방법
from multiprocessing import Pool 
import time
import os

def f(x):
    print('값', x, '에 대한 작업 pid', os.getpid()) # 현재 진행중인 processId를 반환
    time.sleep(1)
    return x * x

if __name__ == '__main__':
    p = Pool(3) # Pool 객체 : 3 ~ 5가 적당
    startTime = int(time.time())
    
    '''
    # 멀티프로세스 사용하지 않고 작업
    for i in range(0, 10):
        print(f(i)) # 10초 소요
    '''
    # 멀티프로세싱을 사용하여 작업 ( Pool )
    # 4초 소요 (프로세스 3개로 작업시)
    print(p.map(f, range(0, 10)))   # 함수(f)와 인자값을 매핑하면서 데이터를 분배 처리
    
    
    
    endTime = int(time.time())
    print('총 작업 시간 : ', (endTime-startTime))
    