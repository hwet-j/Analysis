# 클래스 상속 연습
class Person:
    say = '난 사람이야'
    nai = 20
    __kbs = '공영방송'  # __변수명 : private 멤버, 현재 클래스 내에서만 참조 가능
    
    def __init__(self, nai):
        print('Person 생성자')
        self.nai = nai
        
    def printInfo(self):
        print('나이:{}, 이야기:{}'.format(self.nai, self.say))
        
    def hello(self):
        print('핼로우')
        print('hello : ', self.say, self.__kbs)
        
pe = Person(22)
pe.printInfo()
pe.hello()

print('----------Employee-----------')
class Employee(Person):
    say = '일하는 동물'  # 자식과 동일한 멤버변수에 의해 부모의 say가 숨겨짐
    subject = '근로자' # Employee 고유 멤버 변수
    
    def __init__(self):
        print('Employee 생성자')
        
    def printInfo(self):    # 자식과 동일한 메소드에 의해 부모의 메소드가 숨겨짐
        print('Employee의 printInfo 메소드')
        
    def empShow(self):
        say= '수다 떨기'
        print(say)
        print(self.say)
        print(super().say)
        self.printInfo()
        super().printInfo()
        
        
        
emp = Employee()
print(emp.say, emp.nai)
print(emp.subject)
emp.printInfo()
print()
emp.empShow()

print('----------Worker-----------')
'''
class Worker(Person):
    pass

wo = Worker(33)
print(wo.say, wo.nai)
wo.printInfo()
'''
class Worker(Person):
    def __init__(self,nai):
        print('Worker 생성자 ~~~~~')
        super().__init__(nai)   # Bound call
        # Person.__init__(self, nai) # Unbound call
        
    def woShow(self):
        self.printInfo()
        super().printInfo()
        
        
wo = Worker(27)
wo.woShow()

print('----------Programmer-----------')
class Programmer(Worker):
    def __init__(self, nai):
        print('Programmer 생성자 ~~~~')
        Worker.__init__(self, nai)
        
    def prShow(self):
        self.printInfo()
        super().printInfo()
        
    def kbsShow(self):
        print(self.say)
        print(self.__kbs)
        
pr = Programmer(35)
print(pr.say, pr.nai)
pr.prShow()
pr.hello()
        
print('\n클래스 타입 확인 ')
a = 3
print(type(a))
print(type(pr))

# 부모 클래스 확인
print(Programmer.__bases__)  
print(Worker.__bases__)
print(Person.__bases__)


