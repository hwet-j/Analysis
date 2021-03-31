# Socket : TCP/IP protocol(통신규약)의 프로그래머 인터페이스

import socket

print(socket.getservbyname('http', 'tcp'))      # http 80 기본 port번호
print(socket.getservbyname('telnet', 'tcp'))    # http 23
print(socket.getservbyname('ftp', 'tcp'))       # http 21
print(socket.getservbyname('smtp', 'tcp'))      # http 25
print(socket.getservbyname('pop3', 'tcp'))      # http 110

print()
print(socket.getaddrinfo('www.naver.com', 80, proto=socket.SOL_TCP))
print(socket.getaddrinfo('www.daum.net', 80, proto=socket.SOL_TCP))






