# 스레드를 이용해 날짜 및 시간 출력

import time
import threading

now = time.localtime()
print(now)
print('현재는{0}년{1}월{2}일 {3}시{4}분{5}초'.format(now.tm_year, now.tm_mon, now.tm_mday,\
                                            now.tm_hour, now.tm_min, now.tm_sec))

def cal_show():
    now = time.localtime()
    print('현재는{0}년{1}월{2}일 {3}시{4}분{5}초'.format(now.tm_year, now.tm_mon, now.tm_mday,\
                                            now.tm_hour, now.tm_min, now.tm_sec))

'''
def myRun():
    while True:
        now2 = time.localtime()
        if now2.tm_min == 57:
            break
        cal_show()
        time.sleep(1)
        
th = threading.Thread(target=myRun)
th.start()

th.join()
'''

class MyThread(threading.Thread):    
    def run(self):
        for i in range(1, 10):
            print('id{} ---> {}'.format(self.getName(), i))
            time.sleep(0.1)
            
ths = []
for i in range(3):
    th = MyThread()
    th.start()
    ths.append(th)
    
for th in ths:
    th.join()
    
    
print('프로그램 종료')

        


