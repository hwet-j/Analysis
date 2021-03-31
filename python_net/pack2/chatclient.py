# 멀티 채팅 클라이언트 

import socket
import threading
import sys

def Handle(socket):
    while True:
        data = socket.recv(1024)
        if not data:continue    # 데이터가 존재하지않으면 while문 내에서 continue이후에 문법들을 실행하지않고 다시 while문으로 
        print(data.decode('utf-8'))     # 파이썬의 표준 출력은 버퍼링이 된다.
        
sys.stdout.flush()  # flush(버퍼에대한 클리어작업)를 해주지않고 작업을 해주면 데이터가 넘어오지않을때도 있다. 

cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 소캣생성
cs.connect(('127.0.0.1', 5000)) # connect는 튜플 타입으로 줘야함.(괄호를 하나더주는이유)
name = input('채팅 아이디 입력 :')
cs.send(name.encode('utf-8'))

th = threading.Thread(target = Handle, args = (cs,))       
th.start()

while True:
    msg = input()   # 채팅 메시지 입력
    sys.stdout.flush()
    if not msg:continue    
    cs.send(msg.encode('utf-8'))
    
cs.close()