# matplotlit : ploting library. 그래프(차트) 생성을 위한 다양한 함수 지원
import numpy as np
import matplotlib.pyplot as plt

# 폰트 설정으로 한글깨짐 방지
plt.rc('font', family = 'malgun gothic')
# 한글깨짐 방지이후 음수(-) 깨짐방지    <- 위에 한글깨짐 방지를 사용해서 생김
plt.rcParams['axes.unicode_minus'] = False

'''
x = ["서울","인천","수원"]
y = [5, 3, 7]
plt.xlabel("지역") # x축의 명칭
plt.xlim([-1, 3])   # 축의 값에 대한 제한을 걸어줌
plt.yticks(list(range(0, 11, 3)))   # y축에대한 보여지는 좌표값 설정
plt.ylim([0, 10])
plt.plot(x, y)
plt.show()
'''

#----------------------------
'''
data = np.arange(1, 11, 2)
print(data) #[1 3 5 7 9]
plt.plot(data)  # y축이 값이됨 
x = [0,1,2,3,4]
for a,b in zip(x, data):
    plt.text(a, b, str(b))
plt.show()

plt.plot(data)
plt.plot(data, data, 'r')
plt.show()
'''

'''
x = np.arange(10)
y = np.sin(x)
print(x, y)
#plt.plot(x, y, 'bo')    # style 옵션 적용
#plt.plot(x, y, 'r+')
plt.plot(x, y, 'go--', linewidth = 2, markersize= 10)
plt.show()
'''

'''
# 홀드 : 그림 겹쳐 보기
x = np.arange(0, np.pi * 3, 0.1)
#print(x)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin, 'r')
plt.scatter(x, y_cos)
plt.xlabel('x축')
plt.ylabel('y축')
plt.legend(['사인','코사인'])
plt.title('차트제목')
plt.show()
'''


# subplot : Figure를 여러 행열로 분리
x = np.arange(0, np.pi * 3, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('사인')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('사인')

plt.show()

irum = ['a','b','c','d','e']
kor = [80, 50, 70, 70, 90]
eng = [60, 70, 80, 70, 60]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'gs-')
plt.ylim(0,100)
plt.legend(['국어', '영어'], loc = 4)
plt.grid(True)

fig=plt.gcf()   # 이미지 저장 준비
plt.show()
fig.savefig('test.png') # 실제로 사진을 저장

from matplotlib.pyplot import imread
img = imread('test.png')
plt.imshow(img)
plt.show()











