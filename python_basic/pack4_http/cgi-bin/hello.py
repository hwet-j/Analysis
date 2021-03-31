# 웹 서비스가 가능한 파이썬 파일 작성
ss = '파이썬 변수'
kbs = 9
mbc = 10 + 1.0

print('Content-Type:text/html;charset=utf-8\n')
print('<html>')
print('<body>')
print('<h2>파이썬 문서로 정보 전달</h2>')
print('<b><i>안녕하세요 </i>반가워요</b> 날씨가 춥네요')
print('<br>파이썬 변수 출력 : %s %d %f'%(ss, kbs, mbc))
print('</body>')
print('</html>')
