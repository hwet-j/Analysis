# 클래스

class Car:
    handle = 0      # 멤버 변수(필드), 클래스내에서 전역 
    speed = 0
    
    def __init__(self, name, speed):    # 생성자 오버로딩 X
        self.name = name
        self.speed = speed
        
    def showData(self): # 메소드 오버로딩 X
        km = '킬로미터'    # 지역 변수
        msg = '속도 :' + str(self.speed) + km
        return msg
    
car1 = Car('tom', 10)
print(car1.handle, car1.name, car1.speed)   # 0 tom 10 
car1.color = '핑크'
print('car1.color :', car1.color)

print('car2---------')
car2 = Car('james', 30)
print(car2.handle, car2.name, car2.speed) # 0 james 30
# print('car2.color :', car2.color)   # AttributeError: 'Car' object has no attribute 'color'

print(car1.__dict__)    # {'name': 'tom', 'speed': 10, 'color': '핑크'}
print(car2.__dict__)    # {'name': 'james', 'speed': 30}

print('method --------------')
print('car1 : ', car1.showData())
print('car2 : ', car2.showData())
car1.speed = 100
car2.speed = 200
print('car1 : ', car1.showData())
print('car2 : ', car2.showData())

print()
print(Car.speed)
print(car1.speed)
print(car2.speed)

# print(Car.color)    # AttributeError: type object 'Car' has no attribute 'color'
print(car1.color)
# print(car2.color)   # AttributeError: 'Car' object has no attribute 'color'











