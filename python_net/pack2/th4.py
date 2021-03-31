# 스레드의 활성화 / 비활성화

import threading, time

bread_plate = 0 # 빵 접시 - 공유자원

lock = threading.Condition()

class Maker(threading.Thread):  # 빵 생산자
    def run(self):
        global bread_plate
        
        for i in range(30):
            lock.acquire()  # 공유자원 충돌 방지
            
            while bread_plate >= 10:
                print('빵 생산 초과로 대기')
                lock.wait() # 스레드 비활성화
                
            bread_plate += 1
            print('빵 생산 후 접시에 담기 : ', bread_plate)
            
            lock.notify()   # 스레드의 활성화
            lock.release()
            time.sleep(0.05)
            
class Eater(threading.Thread):  # 빵 소비자
    def run(self):
        global bread_plate
        
        for i in range(30):
            lock.acquire()  # 공유자원 충돌 방지
            
            while bread_plate < 1:
                print('빵이 없어 기다림')
                lock.wait() # 스레드 비활성화
                
            bread_plate -= 1
            print('빵 소비 후 접시에 담기 : ', bread_plate)
            
            lock.notify()   # 스레드의 활성화
            lock.release()
            time.sleep(0.07)        

mak = []
con = []

for i in range(5):  # 빵 생산자 수    
    mak.append(Maker())
    
for i in range(5):  # 빵 소비자 수    
    con.append(Eater())
    
for th1 in mak:
    th1.start()
    
for th2 in con:
    th2.start()
    
# -----------------------------
for th1 in mak:
    th1.join()
    
for th2 in con:
    th2.join()
    
    
print('오늘 장사 끝')    
    