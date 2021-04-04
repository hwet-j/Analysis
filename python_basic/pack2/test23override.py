# 메소드 오버라이딩 : 부모 클래스의 메소드와 동일한 이름의 메소드를 자식 클래스에서 만듦(재정의)
# 다형성의 근거를 제시 

class Parent:
    def printData(self):
        pass
    
    def displayData(self):
        print('Parent의 displayData')
     
        
class Child1(Parent):
    def printData(self):
        a = 10
        b = 20
        print(str(a + b) + '만세 ')
        
class Child2(Parent):
    def printData(self):    # override
        print('Child2 화이팅')
         
    def displayData(self):  # override
        print('Child2의 display : 부모 메소드를 재정의')
        
    def c2method(self):     # Child2의 고유 메소드
        print('Child2 : method')

c1 = Child1()
c1.printData()
c1.displayData()
print()
c2 = Child2()
c2.printData()
c2.displayData()

print('다형성 ')
par = Parent()      # 자바와 비슷한면이 있지만 같지않음
par = c1 # 자식 개체의 주소를 치환
par.printData()
par.displayData()
print()
par = c2 # 자식 개체의 주소를 치환
par.printData()
par.displayData()
par.c2method()

print('\n자바와 다르게 생각할것.')
sbs = c1
sbs.printData()
sbs.displayData()
print()
sbs = c2
sbs.printData()
sbs.displayData()
sbs.c2method()

print()
plist = [c1, c2]
for i in plist:
    i.printData()