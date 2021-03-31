# 챗봇용 서버
from http.server import CGIHTTPRequestHandler, HTTPServer

PORT = 8080

class HandlerClass(CGIHTTPRequestHandler):
    cgi_directories = ['/cgi-bin']
    
serv = HTTPServer(('127.0.0.1', PORT), HandlerClass)    # HTTPServer 서버 객체 생성

print('챗봇용 HTTPServer 서비스 시작...')
serv.serve_forever()    # HTTPServer 서비스 시작    
    
    