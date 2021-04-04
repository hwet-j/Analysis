# 반복문 for
for i in [1, 2, 3, 4, 5]:
    print(i, end = ' ')
    
print()
for i in (1, 2, 3, 4, 5):
    print(i, end = ' ')

print()
for i in {1, 2, 3, 4, 5}:
    print(i, end = ' ')

print()
soft = {'java':'웹용','python':'만능 언어','javascript':'컨턴츠 제약용'}
for i in soft.items():
    # print(i) # ('java', '웹용')
    print(i[0] + '^^;' + i[1])
    
print()
for k, v in soft.items():   
    print(k)
    print(v)
    
print()
for k in soft.keys():
    print(k, end = ' ')
    
print()
for k in soft.values():
    print(k, end = ' ') 
    
print()
for n in [2, 3]:
    print('--{}단--'.format(n))
    for su in [1,2,3,4,5,6,7,8,9]:
        print('{0} * {1} = {2}'.format(n, su, n * su), end='  ')
    print()

print()
# li = ['a', 'b', 'c']     
# li = ('a', 'b', 'c')     
li = {'a', 'b', 'c'}     
for ind, data in enumerate(li):
    print(ind, data)
    
print('continue, break -----')
datas = [1,2,3,4,5]
for i in datas:
    if i == 3:
        # continue
        break
    print(i, end = ' ')
else:
    print('정상 처리')
    
print('계속')
jumsu = [95, 70, 60, 50, 100]
number = 0

for jum in jumsu:
    number += 1
    if jum < 70: continue
    print('%d번째 점수는 합격'%number)
    
print()
li1 = [3, 4, 5]    
li2 = [0.5, 1, 2]
for a in li1:
    for b in li2:
        print(a + b, end = ' ')
        
print()
print([a + b for a in li1 for b in li2])
datas = [a + b for a in li1 for b in li2]
for d in datas:
    print(d, end = ' ')
    
print('쉬어가기 ---------')    
import re
ss = '''중앙방역대책본부(방대본)는 9일 0시 기준으로 국내 코로나19 신규 확진자가 303명 늘었다고 밝혔다. 
누적 확진자는 8만1487명을 기록했다.
누적 확진자는 8만1487명을 기록했다.
누적 확진자는 8만1487명을 기록했다.
코로나19 신규 확진자는 전날(289명)보다 14명 늘며 200명대로 떨어진 지 하루 만에 다시 300명대로 올라섰다.
누적 확진자는 8만1487명을 기록했다.'''

ss2 = re.sub(r"[^가-힣\s]", "", ss)
print(ss2) 
ss3 = ss2.split(' ')  
print(ss3)

cou = {} #  단어의 발생횟수를 dict로 저장
for n in ss3:
    if n in cou:
        cou[n] += 1
    else:
        cou[n] = 1

print(cou)

print()
for ss in ['111-1234', '일이삼-사오육칠', '123-1234']:
    if re.match(r'\d{3}--\d{4}', ss):
        print(ss, '전화번호 맞아요')
    else:
        print(ss, '전화번호 아닌데')
    
print()
from time import localtime
print(localtime())
act = {6:'참', 9:'아침먹고 수업준비', 18:'공부', 24:'휴식'}
print(act)
hour = localtime().tm_hour

for act_time in sorted(act.keys()):
    if hour < act_time:
        print(act[act_time])
        break
    #else:
    #    print('너는 뭐야')
    
print('과일 값 계산하기')
price = {'사과':3000, '감':500, '한라봉':1000}
guest = {'사과':2, '감':3}
bill = sum(price[f] * guest[f] for f in guest)
print('손님이 구매한 과일 값 총액은 {}원'.format(bill))

print()
datas = [1, 2, 'kbs', True, 3]
li = [i * i for i in datas if type(i) == int]
print(li)

datas = {1,1,2,2,3}
se = {i * i for i in datas}
print(se)

id_name = {1:'tom',2:'john'}
print(id_name)
name_id = {val:key for key, val in id_name.items()}
print(name_id)

print()
aa = [(1, 2), (3, 4), (5, 6)]
for a, b in aa:
    print(a+b)
    
temp = [1, 2, 3]
temp2 = [i + 10 for i in temp]
print(temp2)

print('\n수열 생성 - range()')
print(list(range(1, 6)))
print(set(range(1, 6)))
print(tuple(range(1, 6)))
print(list(range(1, 11, 3)))
print(list(range(-10, -200, -30)))

for i in range(6):
    print(i, end = ' ')

print()
tot = 0
for i in range(1, 50, 2):
    print(i, end = ' ')
    tot += i
print()
print('결과는' + str(tot))
print('결과는', sum(range(1, 50, 2)))

            
print('구구단 2~5단')
for i in range(2, 6):
    for j in range(1, 10):
        print('{} * {} = {}'.format(i, j, i*j), end=' ')
    print()

print('1 ~ 100 사이의 숫자중 3의 배수이면서 5의 배수인 수들의 합 출력')    
tot = 0
for i in range(1, 101):
    # print(i, end = ' ')
    if i % 3 == 0:
        tot = tot + i
    
print('합은', tot)

# N-gram : 문자열에서 n개의 연속된 요소를 추출하는 방법
# 글자별 n-gram : 2-gram 
ss = 'hello'

for i in range(len(ss)-1):
    print(ss[i], ss[i+1], sep=' ')

print()
# 단어별 n-gram : 2-gram
ss2 = 'this is python script'
words = ss2.split()

for i in range(len(words)-1):
    print(words[i], words[i+1], sep=' ')
    
    
    
    