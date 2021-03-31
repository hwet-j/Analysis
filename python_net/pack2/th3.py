# 스레드 간 공유자원 값 충돌 방지

import threading, time

g_count = 0 # 전역변수는 자동으로 스레드의 공유자원이 됨

lock = threading.Lock()

def threadCount(id, count):
    global g_count
    
    for i in range(count):
        lock.acquire() # 두 개 이상의 스레드 공유자원 충돌방지. lock을 걸어 다른 수행 대기
        
        print('id %s ==> count : %s g_count:%s'%(id, i, g_count))
        g_count += 1
        lock.release()  # lock을 해제
        
for i in range(1, 6):
    threading.Thread(target=threadCount, args=(i, 5)).start()
    
time.sleep(2)
print('final g_count : ', g_count)
print('finish process')
