# 사용자 정의 모듈 
tot = 123 # 전역 변수

def ListHap(*ar):
    print(ar)
    
    if __name__ == '__main__':
        print('응용 프로그램이 시작되는 모듈')

def kbs():
    ch = 9
    print('공영방송 :', ch)
    
def mbc():
    print('문화방송')