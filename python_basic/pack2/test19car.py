# Car 클래스 : 여러 개의 부품(클래스)을 조립해서 완성차를 생성
from pack2.test19handle import PohamHandle

class PohamCar:
    speed = 0
    turnShow = '정지'
    
    def __init__(self, ownerName):
        self.ownerName = ownerName
        self.handle = PohamHandle()     # 클래스의 포함관계
        
    def TurnHandle(self, q):    # PohamCar 메소드로 핸들을 움직이는 행위
        if q > 0:
            self.turnShow = self.handle.RightTurn(q)
        elif q < 0:
            self.turnShow = self.handle.LeftTurn(q)
        elif q == 0:
            self.turnShow = '직진'
            
if __name__ == '__main__':
    tom = PohamCar('톰')
    tom.TurnHandle(10)
    print(tom.ownerName + '의 회전량은' + tom.turnShow + str(tom.handle.quantity))
    tom.TurnHandle(0)
    print(tom.ownerName + '의 회전량은' + tom.turnShow)
    
    print()
    oscar = PohamCar('오스카')
    oscar.TurnHandle(-5)
    print(oscar.ownerName + "의 회전량은" + oscar.turnShow + str(oscar.handle.quantity))
    
