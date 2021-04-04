# 2번문제
# https://cafe.daum.net/flowlife/RUrO/24
# 2. 클래스의 상속관계 연습문제 - 다형성 


class ElecProduct:
    volume = 0
    
    def volumeControl(self, volume):
        pass
        
class ElecTv(ElecProduct): 
    def volumeControl(self, volume):
        print('ElecTv의 volume :', volume)
    
    
class ElecRadio(ElecProduct):
    def volumeControl(self, volume):
        print('ElecRadio의 volume :', volume)
        print(super().volume)

# 출력해보기        
show1 = ElecTv()
show1.volumeControl(40)

show2 = ElecRadio()
show2.volumeControl(20)

