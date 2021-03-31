# 일회 접속용 클라이언트 소스 코드

from socket import *

clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('127.0.0.1', 9999))
clientSock.sendall('안녕 반가워'.encode(encoding='utf_8', errors='strict'))
re_msg = clientSock.recv(1024).decode()

