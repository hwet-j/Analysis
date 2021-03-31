# CGI(Common Gateway Interface)란 웹서버(정보제공측)와 클라이언트(정보이용측)간에 
# 필요한 정보교환을 가능하게 해주는 일종의 웹인터페이스라고(일종의 프로그램) 할 수 있습니다. PHP, ASP, ASP.NET, JSP ....

# CGI를 지원하는 CGIHTTPRequestHandler를 사용하면 클라이언트와 서버 사이에 자료를 주고 받을수 있다. py문서를 처리가능
# get, post 요청 처리 모두 지원

from http.server import CGIHTTPRequestHandler, HTTPServer

PORT = 8888

class HandlerClass(CGIHTTPRequestHandler):
    cgi_directories = ['/cgi-bin']
    
serv = HTTPServer(('127.0.0.1', PORT), HandlerClass)    # HTTPServer 서버 객체 생성

print('HTTPServer 서비스 시작...')
serv.serve_forever()    # HTTPServer 서비스 시작    
    
    