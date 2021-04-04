# 커피 자판기 

class CoinIn:
    coin = 0
    change = 0
    
    def calc(self, cupCount):
        re = ""
        if self.coin < 200:
            re = '요금 부족'
        elif cupCount * 200 > self.coin:
            re = '요금 부족'
        else:
            self.change = self.coin - (200 * cupCount)
            re = "커피" + str(cupCount) + '잔과 잔돈 ' + str(self.change) + "원"
        
        return re

class Machine():
    cupCount = 1
    
    def __init__(self):
        self.coinIn = CoinIn()  # 포함
        
    def showData(self):
        self.coinIn.coin = int(input("동전을 입력하세요"))
        self.cupCount = int(input("몇 잔을 원하십니까"))
        print(self.coinIn.calc(self.cupCount))
        
machine = Machine()
machine.showData()
