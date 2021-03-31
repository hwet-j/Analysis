#  문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("../testdata/cleanDescriptive.csv")
df = data.dropna(axis=0)
print(df)
df2 = data[['level', 'pass']].dropna(axis=0)
print(df2)

# 집단 : 부모학력, 자녀진학
# 귀무 : 부모학력과 자녀진학 간의 관련이 없다. (독립)
# 대립 : 부모학력과 자녀진학 간의 관련이 있다. (독립이 아니다.)

# 부모학력, 자녀진학을 이용해 교차표 작성
ctab = pd.crosstab(index = df['level'], columns= df['pass'])
ctab2 = pd.crosstab(index = df2['level'], columns= df2['pass'])
ctab.index = ['고졸','대졸','대학원졸']
ctab.columns = ['합격','실패']
print(ctab)

chi_result = [ctab.loc['고졸'], ctab.loc['대졸'], ctab.loc['대학원졸']]
chi, p, _, _= stats.chi2_contingency(chi_result)
chi2, p2, _, _= stats.chi2_contingency(ctab2)
# 전체 데이터에서 NA값존재 제외
print('chi :',chi)      # chi : 37.403493
print('p-value :',p)    # p-value : 0.0202980
# level,pass 데이터에서 NA값존재 제외
print('chi2 :',chi2)      # chi : 2.7669512
print('p2-value :',p2)    # p-value : 0.25070568

# 해석1 : p-value : 0.0202980 < 0.05 이므로 귀무가설을 기각
# 대립 : 부모학력과 자녀진학 간의 관련이 있다. (독립이 아니다.)

# 해석2 : p-value : 0.25070568 > 0.05 이므로 귀무가설
# 귀무 : 부모학력과 자녀진학 간의 관련이 없다. (독립)
