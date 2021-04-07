# 크롤링한 데이터를 읽어와 형태소 분석해보기..

import tensorflow as tf
import numpy as np 
import nltk
from konlpy.tag import Okt
import pandas as pd
import rhinoMorph   # pip install rhinoMorph

data = pd.read_csv("Twitter_all_data.txt", "\t")
print(data.head(3))
print(data.isnull().any())

okt = Okt()
print(okt.pos(u'흔들리는 꽃들 속에서 네 샴푸향이 느껴진거야'))