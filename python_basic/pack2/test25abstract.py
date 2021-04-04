# 추상 클래스 : 추상 메소드를 갖음 - 자식 클래스에서 부모의 추상메소드를 일반메소드로 반드시 오버라이딩 하도록 강요.
# 다형성을 구사하기 위해서 추상클래스가 사용되는 것임.

from abc import *

class TestClass(metaclass = ABCMeta):   # 추상 클래스
    @abstractmethod
    def abcMethod(self):    # 추상 메소드 - 추상 메소드는 오버라이딩을 해주지 않으면 오류
        pass
    
    def normalMethod(self):
        print('일반메소드')
        
# 추상클래스가 되고 추상메소드가 만들어짐으로써 class자체에서 메소드를 사용할수 없다.
# tt = TestClass()    # TypeError: Can't instantiate abstract class TestClass with abstract methods abcMethod

class Child1(TestClass):
    name = '난 Child1'
    
    def abcMethod(self):
        print('추상 메소드를 오버라이딩해서 일반메소드가 됨')
    
c1 = Child1()
c1.abcMethod()
c1.normalMethod()

class Child2(TestClass):
    name = '난 Child2'
    
    def abcMethod(self):
        print('-----------')
        print('Child2에서 추상 메소드를 오버라이딩해서 일반메소드가 됨')
        print('다른 클래스의 abcMethod와 이름은 같으나 다른 기능을 수행')
        print('-----------')
     
    def normalMethod(self):
        print('부모의 normalMethod를 오버라이딩함 부모와 역할이 다른 메소드를 수행하기 위해')    
        
print()        
c2 = Child2()
c2.abcMethod()
c2.normalMethod()        
        
print('\n다형성 -----------')       
my = c1
my.abcMethod()  # 1

my = c2
my.abcMethod()  # 1과 동일한 명령문이지만 기능은 다르다.

        
print('===========카페 python.24.추상연습==================')        
# from abc import * # 위에서 작성했기때문에 필요없음 원래는 필요함.
class Employee(metaclass = ABCMeta):
    def __init__(self, irum, nai):
        self.irum = irum
        self.nai = nai
        
    @abstractmethod
    def pay(self):
        pass

    @abstractmethod
    def data_print(self):
        pass
    
    def irumnai_print(self):
        print('이름 : ' + self.irum + ', 나이  : ' + str(self.nai), end = ' ')
        
class Temporary(Employee):
    def __init__(self, irum, nai, ilsu, ildang):
        Employee.__init__(self, irum, nai)
        self.ilsu = ilsu
        self.ildang = ildang
        
    def pay(self):
        return self.ilsu * self.ildang
    
    def data_print(self):
        self.irumnai_print()
        print(', 월급 : ' + str(self.pay()))
        
t = Temporary('홍길동', 24, 20, 15000)
t.data_print()
        
class Regular(Employee):
    def __init__(self, irum, nai, salary):
        Employee.__init__(self, irum, nai)
        # super().__init__(irum, nai) # 이렇게 작성해줘도 됨
        self.salary = salary
        
    def pay(self):  # 추상 메소드를 오버라이딩
        return self.salary
    
    def data_print(self):    # 추상 메소드를 오버라이딩
        self.irumnai_print()
        print(', 급여 : ' + str(self.pay()))
        
r = Regular('한국인', 27, 3500000)
r.data_print()   

class Salesman(Regular):     
    def __init__(self, irum, nai, salary, sales, commission):
        super().__init__(irum, nai, salary)
        self.sales = sales
        self.commission = commission
        
    def pay(self):  # 일반 메소드를 오버라이딩
        return super().pay() + (self.sales * self.commission)
    
    def data_print(self):    # 일반 메소드를 오버라이딩
        self.irumnai_print()
        print(', 수령액 : ' + str(self.pay()))
        
        
s = Salesman('손오공', 29, 1200000, 5000000, 0.25)
s.data_print()         
        
        
        