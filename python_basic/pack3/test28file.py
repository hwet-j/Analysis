# 동 이름으로 우편번호와 주소 출력

try:
    dong = input('동 이름 입력 : ')
    print(dong)
    
    with open(r'zipcode.txt', mode = 'r', encoding='euc-kr') as f:
        #print(f.read())
        line = f.readline() 
        
        while line:
            lines = line.split('\t')
            # print(lines)    # ['135-806', '서울', '강남구', '개포1동 경남아파트', '', '1\n']
            if lines[3].startswith(dong):   # 입력된 동이름으로 시작되는 주소만 작업
                # 명령문이 너무 길어질때 백슬레쉬(\) 를 사용해서 줄바꿈을 해준다.
                print('[' + lines[0] + '] ' + lines[1] + ' ' + lines[2] + ' '  + \
                      lines[3] + ' ' + lines[4])
            line = f.readline()

except Exception as e:
    print("에러 : ", e)