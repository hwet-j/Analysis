# process : 실행 중인 응용프로그램을 의미. 프로세스 단위로 별도의 메모리를 사용
'''
from subprocess import *

Popen('calc.exe')
Popen('notepad.exe')
'''

# thread : Light weight process라고도 함. 메인 프로세스(스레드)와 병령적으로 수행되는 단위 프로그램. 스레드 단위로 함수나 메소드를 수행 가능
import time


def run(id):
    for i in range(1, 11):
        print('id:{} --> {}'.format(id, i))
        time.sleep(0.5)

        
# 1. thread를 사용하지 않은 경우
# run(1)  # 순차적으로 호출되므로 순차적으로 처리됨
# run(2)

import threading

# 2. thread를 사용하는 경우 : 스레드 스케줄려에 의해 랜덤하게 스레드 처리가 됨
th1 = threading.Thread(target=run, args = ('일'))    # 사용자 정의 쓰레드 생성 
th2 = threading.Thread(target=run, args = ('둘'))
th1.start() # 스레드 수행 시작
th2.start()
th1.join()  # 사용자 정의 스레드가 끝날때 까지 메인 스레드를 대기 
th2.join()

print('메인 프로그램 종료') # 메인 스레드에 의해 메인 모듈이 실행 