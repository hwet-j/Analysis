# 일반적으로 test20 의 방식보다는 test19 방식을 선호한다.
# 냉장고(class)에 음식(class)을 저장

class FoodData:
    def __init__(self, irum, yutong):
        self.irum = irum
        self.yutong = yutong
        
        
class Fridge:
    isOpened = False    # 냉장고 문 개폐 여부
    foods = []
    
    def open(self):
        self.isOpened = True
        print('냉장고 문 열기')
        
    def close(self):
        self.isOpened = False
        print('냉장고 문 닫기')
        
    def put(self, thing):
        if self.isOpened:
            self.foods.append(thing)    # 클래스의 포함
            print('냉장고 속에 음식을 담음')
            self.listFood()
        else:
            print('냉장고 문이 닫혀있어 음식을 담을수 없어요.')
        
    def listFood(self):
        for f in self.foods:
            print('-', f.irum, f.yutong)
            
f = Fridge()
apple = FoodData('사과', '2021-3-5')
f.put(apple)
f.open()
f.put(apple)
f.close()

cola = FoodData('콜라', '2021-3-5')
f.open()
f.put(cola)
f.close

        
        