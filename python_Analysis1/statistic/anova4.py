# 이원 분산분석 : 요인 2개
import scipy.stats as stats
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 강남구 소재 GS 편의점 3개지역, 알바생의 급여에 대한 평균의 차이를 검정
# 귀무 : 3개 지역 급여에 대한 평균에 차이가 없다.
# 대립 : 3개 지역 급여에 대한 평균에 차이가 있다.

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(urllib.request.urlopen(url))
print(data.head(5))

# 귀무 : 태아수와 관측자수는 태아의 머리둘레와 관련이 없다. (머리둘레 길이의 평균)
# 대립 : 태아수와 관측자수는 태아의 머리둘레와 관련이 있다.

#data.boxplot(column='머리둘레', by='태아수', grid=False)
#plt.show()

reg = ols('data["머리둘레"] ~ C(data["태아수"]) + C(data["관측자수"])', data = data).fit()
result = anova_lm(reg, type = 2)
print(result)

# 두 개의 요소 상호작용이 있는 형태로 처리
formula = "머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)"
reg2 = ols(formula, data).fit()
result2 = anova_lm(reg2, type = 2)
print(result2)  # p-value : 3.295509e-01 > 0.05 이므로 귀무가설 채택

