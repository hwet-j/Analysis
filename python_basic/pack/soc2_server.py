# 서버 서비스 계속 유지
import socket
import sys

HOST = ''       
PORT = 8888

serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    serverSock.bind((HOST, PORT))
    print('server start...')
    serverSock.listen(5)    # 클라이언트 접속 대기. 동시 접속 최대수는 1~5(접속이 5명 까지가 아니라 동시에 접속이 5명으로 제한임 )
    
    while(True):
        conn, addr = serverSock.accept()    # 클라이언트 접속 승인
        print('client info :', addr[0], addr[1])
        print(conn.recv(1024).decode()) # 클라이언트가 전송한 메시지 수신후 출력
        
        # 전송    - 보낼때 encode, 받을때 decode
        conn.send(('from server :' + str(addr[1]) + '너도 파이팅').encode('utf_8'))
except socket.error as err:
    print('soc err : ', err)
except Exception as e:
    print('err : ', e)
finally:
    serverSock.close()