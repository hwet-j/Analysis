# file i/o

import os
print(os.getcwd())

try:
    print('파일 읽기')
    f1 = open(r'ftest.txt', mode='r', encoding='utf-8')
    print(f1)
    print(f1.read())
    f1.close()  # 파일 처리 종료시 close() : 사용했던 자원을 해제
    
    print('파일로 저장')
    f2 = open(r'ftest2.txt', mode='w', encoding='utf-8')
    f2.write('내 친구들\n')
    f2.write('홍길동, 신기해, ')
    f2.write('홍삼, 신선해')
    f2.close()
    print('저장 성공')
    
    print('파일로 내용 추가')
    f2 = open(r'ftest2.txt', mode='a', encoding='utf-8')
    f2.write('손오공\n')
    f2.write('조조')
    f2.close()
    print('추가 성공')
    
    f3 = open(r'ftest2.txt', mode='r', encoding='utf-8')
    print(f3.read())    # 전체행 읽기
    # print(f3.readline())    # 한행씩 읽기 - 전체출력시 반복문
    f3.close() 
    
except Exception as e:
    print('에러  : ', e)
    
    
print('\n------with문 사용 - close() 자동 수행 -------------')    
try:
    # 저장
    with open('ftest3.txt', mode = 'w', encoding = 'utf-8') as ff1:
        ff1.write('파이썬 문구 저장 \n')
        ff1.write('어쩌구 저쩌구\n')
        ff1.write('with문을 사용하니까 close()사용하지 않음\n')
        
    # 읽기
    with open('ftest3.txt', mode = 'r', encoding = 'utf-8') as ff2:
        print(ff2.read())
        
except Exception as e2:
    print('에러  : ', e2)
    
    
print('\n------피클링 (복합개체 처리(i/o)-------------')    

import pickle

try:
    dicData = {'tom':'111-1111','james':'222-2222'}
    listData = {'마우스', '키보드'}
    tupleData = (dicData, listData) # 복합 객체
    
    with open('hi.dat', 'wb') as fobj:
        pickle.dump(tupleData, fobj)
        pickle.dump(listData, fobj)
    print('객체를 파일로 저장하기 성공')
    
    print('객체 읽기')
    with open('hi.dat', 'rb') as fobj2:
        a, b = pickle.load(fobj2)
        print(a)    # {'tom': '111-1111', 'james': '222-2222'}  
        print(b)    # {'마우스', '키보드'}
        c = pickle.load(fobj2)
        print(c)
        
        
except Exception as e3:
    print('에러  : ', e3)








































































