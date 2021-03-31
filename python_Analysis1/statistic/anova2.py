# 일원분산으로 집단 간의 평균 차이 검정
import scipy.stats as stats
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 강남구 소재 GS 편의점 3개 지역 , 알바생의 급여에 대한 평균의 차이를 검정
# 귀무 : 3개 지역 급여에 대한 평균에 차이가 없다.
# 대립 : 3개 지역 급여에 대한 평균에 차이가 있다.

url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt"
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')
print(data)

# 세 개의 집단
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(gr1, np.average(gr1))     # 316.625
print(gr2, np.average(gr2))     # 256.44444444444446
print(gr3, np.average(gr3))     # 278.0

# 정규성
print(stats.shapiro(gr1))
print(stats.shapiro(gr2))
print(stats.shapiro(gr3))

plot_data = [gr1, gr2, gr3]
plt.boxplot(plot_data)
#plt.show()

# 등분산성
print(stats.bartlett(gr1, gr2, gr3))    # pvalue=0.350803 > 0.05

# 방법1 
df = pd.DataFrame(data, columns = ['value','group'])
print(df)
model = ols('value ~ C(group)', df).fit()   # C(변수명 + 변수명2 +......) ==>> 범주형임을 명시적으로 표시
print(anova_lm(model))

print()
# 방법2
f_statistic, p_val = stats.f_oneway(gr1, gr2, gr3)
print('f_statistic : ', f_statistic)
print('p_val : ', p_val)






