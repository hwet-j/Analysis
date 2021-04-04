# Graphic 지원 모듈 사용 - turtle
from turtle import *
pen = Pen()

pen.color('red', 'yellow')
pen.begin_fill()

while True:
    pen.forward(200)
    pen.left(170)
    if abs(pen.pos()) < 1:
        break
pen.end_fill()
done()

