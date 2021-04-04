# class : OOP 기법 구사
# 클래스는 새로운 이름 공간을 지원하는 단위다. 멤버는 변수, 메소드, 생성자로 구성, 접든지정지X, method overloading X

print('이런 저런 작업을 하다가 클래스 등장..')

def func():
    print('함수')

class TestClass : # 원형 클래스 - prototype
    abc = 1 #맴버 변수는 전역
    
    def __init__(self):
        print('생성자')
        
    def __del__(self):
        print('소멸자')
        
    def printMsg(self): # method
        name = '홍길동'    # 지역 변수
        print(name)
        print(self.abc)
        self.show()
        
    def show(self):
        print('show')
        
print(TestClass.abc)    # 원형 클래스의 멤버 변수 호출
# TestClass.printMsg(self)    # NameError : name 'self' is not defined

test = TestClass()  # 생성자 호출된 후 객체 생성(instance)
print(test.abc)
test.printMsg()     # 1. Bound method call
print('--------')
TestClass.printMsg(test)    # 2. UnBound method call
print()
print('클래스 타입 확인 :', type(1))
print('클래스 타입 확인 :', type(test))
print(id(test))
print(id(TestClass))

