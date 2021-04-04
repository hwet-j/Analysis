# 다중 상속 - 순서가 중요 
print("어쩌구 저쩌구")
class Tiger:    # 클래스명 뒤에 괄호내용이 없으면 상속받을것이 없다고 이해
    data = '호랑이 세상'
    
    def cry(self):
        print('호랑이 울부짖음 ')
        
    def eat(self):
        print('맹수는 고기를 먹음')
        
class Lion:
    #data = '사자 세상'    # 우선순위가 Lion일때 이값이 존재하면 data가 이값이 먼저 값이없다면 이후의 값인 Tiger 값으로 넘어감
    def cry(self):
        print('백수의 왕 으르렁')
        
    def hobby(self):
        print('사자의 취미는 낮잠')
        

class Liger(Tiger, Lion):
    pass

aa = Liger() 
aa.cry()    # 상속을 먼저 적어준 데이터가 우선순위가 있다.
aa.eat()
aa.hobby()
print(aa.data)

print('***' * 10)
class Liger2(Lion, Tiger):
    data = '라이거 멤버변수'
    def play(self):
        print('Liger2의 고유 메소드')
        self.hobby()
        super().hobby()
        print(self.data)
        print(super().data)
        
bb = Liger2()
bb.cry()
bb.eat()
bb.hobby()
print(bb.data)
print()
bb.play()

print('/n----------------')
class Animal:
    def move(self):
        pass
    
class Dog(Animal):
    name = '개'
    def move(self):
        print('개는 낮에 돌아다님')
        
class Cat(Animal):    
    name = '고양이'
    def move(self):
        print('고양이는 밤에 움직임')
        print('눈빛이 빛님')
        
class Wolf(Dog, Cat):
    pass

class Fox(Cat, Dog):
    def move(self):
        print('오버라이딩: 나는 여우')
        
    def foxMethod(self):
        print('여우 고유 메소드')
        
dog = Dog()
cat = Cat()
wolf = Wolf()
fox = Fox()

anis = [dog, cat, wolf, fox]
for a in anis:
    a.move()
    



        
