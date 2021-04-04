# class

kor = 100

def abc():
    print('모듈의 멤버 함수')
    
class MyClass:
    kor = 88
    
    '''
    내용이 존재하지 않을때는 아무것도 안적어준것과 동일함.
    def __init__(self):
        pass
    '''
    def abc(self):
        print('클래스의 멤버 메소드임을 선언하노라')
        
    def showData(self):
        # kor = 77
        print(self.kor)
        print(kor)
        self.abc() # 현재 클래스 내의 메소드 콜
        abc() # 모듈의 함수를 콜
        
obj = MyClass()
obj.showData()

print('------------')
class My:
    a = 1
    
print(My.a)
print()
my1 = My # 생성자를 부른게 아닌 주소값을 복사함.
print(my1.a)

print()
my2 = My()
my2.a = 100
print(my2.a)

print()
my3 = My()
my3.a = 200
print(my3.a)
my3.b = 123
print(my3.b)

# print(My.b)
# print(my1.b)
# print(my2.b)


