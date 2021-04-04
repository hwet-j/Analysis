# 클래스는 설계도 
# Singer(가수)가 갖춰야 할 기본 속성, 행위를 설계도(원형 클래스)로 만든 후 모든 가수들은 Singer type으로 존재하면 됨

class Singer:
    title_song = '화이팅 코리아'  # 멤버 변수
    
    def __init__(self):
        pass
    
    def sing(self):
        msg = '노래는'
        print(msg, self.title_song)
        
# 편의상 아래에서 객체를 만들지만 별도의 모듈에서 Singer를 불러다 사용하는 것이 일반적이다.
bts = Singer()
bts.sing()
bts.title_song = '다이너마이트'
bts.sing()
bts.co = '빅히트'
print('소속사', bts.co)

