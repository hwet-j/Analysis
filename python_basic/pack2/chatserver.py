# 멀티 채팅 서버 : socket, thread 사용

import socket
import threading

ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # socket(socket종류, socket유형)
ss.bind(('127.0.0.1', 5000))
ss.listen(5)    # 동시 참여하고있는 인원 5명이 아닌 동시접속 5인
print('채팅 서비스 시작..')
users = []

def ChatUser(conn):
    name = conn.recv(1024)
    data = 'oㅗo' + name.decode('utf-8') + '님 입장~'
    print(data) # 잘되는지 확인
    
    try:
        for p in users:
            p.send(data.encode('utf-8'))    # 각 유저마다 data를 보냄.
        
        while True: # 수다 떨기 
            msg = conn.recv(1024)
            data = name.decode('utf-8') + '님 메시지 : ' + msg.decode('utf-8')
            print('수신 : ', data)
            for p in users:
                p.send(data.encode('utf-8'))
            
    except:
        users.remove(conn)
        data = '~~' + name.decode('utf-8') + '님 퇴장~'
        print(data)
        if users:
            for p in users:
                p.send(data.encode('utf-8'))
        else:
            print('exit')
        
    
while True:
    # 사용자가 들어오면 사용자의 정보가 여기로 먼저 들어온후에 ChatUser에 (conn)으로 들어감.
    conn, addr = ss.accept()    
    users.append(conn)  # 사용자가 들어올때마다 연결정보를 저장함.
    th = threading.Thread(target=ChatUser, args = (conn,))  # argument는 tuple타입으로 줘야함. 유저마다 스레드가 생성됨.
    th.start()      # 호출