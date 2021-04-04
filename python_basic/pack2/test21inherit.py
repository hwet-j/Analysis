# 클래스가 다른 클래스의 멤버를 활용 - 상속 : 다형성을 구사

# --------------------------
class Animal:   # 별도의 모듈에서 작성하고 불러다 썻다고 가정
    age = 0
    
    def __init__(self):
        print('Animal 생성자')
    
    def move(self):
        print('움직이는 생물')
        
#----------------------------

print('어쩌구 저쩌구....')

class Dog(Animal):  # 상속
    age = 10
    
    def __init__(self):
        print('Dog 생성자')
            
    def dogShow(self):
        age = 2
        print('난 멍멍이~~')
        print('age : ', age)
        print('age : ', self.age)
        print('age : ', super().age)
        
dog1 = Dog()
print('dog1.age :',dog1.age)
dog1.move()
print()
dog1.dogShow()

print('-----')
class Horse(Animal):
    pass

horse = Horse()
horse.move()

