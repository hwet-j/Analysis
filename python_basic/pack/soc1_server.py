# 접속상태 확인을 위한 단순 Echo 서버 - client의 요청에 1회만 반응

from socket import *

serverSock = socket(AF_INET, SOCK_STREAM)   # socket(socket종류, socket유형) 
serverSock.bind(('127.0.0.1', 9999))
serverSock.listen(1)    # TCP 리스너 설정

print('server start....')

conn, addr = serverSock.accept()
print('client addr :', addr)
print('from client message : ', conn.recv(1024).decode())   # 클라이언트로부터 바이트 단위 문자열 수신된 내용 출력 

conn.close()
serverSock.close()
