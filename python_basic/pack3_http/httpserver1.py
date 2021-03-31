# HttpServer 구축용 클래스를 이용해 웹서버 서비스 하기 
# HTTPServer : 기본적인 socket 연결을 관리
# SimpleHTTPRequestHandler : get, head 처리 가능


from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 7777

handler = SimpleHTTPRequestHandler
#serv = HTTPServer(('192.168.0.101', PORT), handler) # HTTPServer 서버 객체 생성 
serv = HTTPServer(('127.0.0.1', PORT), handler)
print('HTTPServer 서비스 시작...')
serv.serve_forever()    # HTTPServer 서비스 시작


