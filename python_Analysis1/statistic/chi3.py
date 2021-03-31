"""
이원카이제곱 - 교차분할표 이용
: 두 개 이상의 변인(집단 또는 범주)을 대상으로 검정을 수행한다.
분석대상의 집단 수에 의해서 독립성 검정과 동질성 검정으로 나뉜다.
독립성 검정  : 두 변인의 관계가 독립인지를 결정 
동질성 검정 : 두 변인 간의 비율이 서로 동일한지를 결정

독립성(관련성) 검정
- 동일 집단의 두 변인(학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
- 독립성 검정은 두 변수 사이의 연관성을 검정한다.
실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv'
"""

import pandas as pd
import scipy.stats as stats

data = pd.read_csv("../testdata/smoke.csv")
print(data)
print(data['education'].unique())   # [1 2 3]
print(data['smoking'].unique())     # [1 2 3]

# 집단2 : 교육수준, 흡연율
# 귀무 : 교육수준과 흡연율 간의 관련이 없다. (독립)
# 대립 : 교육수준과 흡연율 간의 관련이 있다. (독립이 아니다.)


# 교육수준, 흡연인원수를 이용해 교차표 작성
ctab = pd.crosstab(index = data['education'], columns= data['smoking'])
#ctab = pd.crosstab(index = data['education'], columns= data['smoking'], normalize=True) # normalize을 True로 주면 비율로 값을 확인가능
ctab.index = ['대학원졸','대졸','고졸']
ctab.columns = ['꼴초','보통','노담']
print(ctab)

# 이원카이제곱을 지원하는 함수
chi_result = [ctab.loc['대학원졸'], ctab.loc['대졸'], ctab.loc['고졸']]
chi, p, _, _= stats.chi2_contingency(chi_result)
#chi, p, _, _= stats.chi2_contingency(ctab)
print('chi :',chi)
print('p-value :',p)

# 해석 : p-value : 0.00081825 < 0.05 이므로 귀무가설을 기각
# 대립 = 교육수준과 흡연율 간의 관련이 있다. (독립이 아니다)

# 참고 : 분할표의 자유도가 1인경우는 x^2값이 약간 높게 계산된다. 그래서 아래의 식과 같이 절대값 |O-E|에서 0.5를 뺸다음 제곱하며,
# 이 방법을 야트보정이라 한다. x^2 = Σ(|O-E| - 0.5)^2 / E
# 자동으로 야트보정이 이루어진다.


print('\n\n\n')
# 실습) 국가전체와 지역에 대한 인종 간 인원수로 독립성 검정 실습
# 두 집단(국가전체 - national, 특정지역 - la)의 인종 간 인원수의 분포가 관련이 있는가?

# 귀무 = 국가전체와 지역에 대한 인종 간 인원수는 관련이 없다. (독립)
# 대립 = 국가전체와 지역에 대한 인종 간 인원수는 관련이 있다 (독립이 아니다)


national = pd.DataFrame(["white"] * 100000 + ["hispanic"] * 60000 +
                        ["black"] * 50000 + ["asian"] * 15000 + ["other"] * 35000)
la = pd.DataFrame(["white"] * 600 + ["hispanic"] * 300 + ["black"] * 250 +
                  ["asian"] * 75 + ["other"] * 150)

print(national)   # [260000 rows x 1 columns]
print(la)   # [1375 rows x 1 columns]

na_table = pd.crosstab(index = national[0], columns = 'count')
la_table = pd.crosstab(index = la[0], columns = 'count')
print(na_table)
print(la_table)

# na_table에 칼럼추가
na_table['count_la'] = la_table['count']
print(na_table)

chi, p, _, _= stats.chi2_contingency(na_table)
print('chi :',chi)  # chi : 18.0995
print('p-value :',p)    # p-value : 0.001180

# 해석 : p-value : 0.001180 < 0.05 이므로 귀무가설을 기각
# 대립 = 국가전체와 지역에 대한 인종 간 인원수는 관련이 있다 (독립이 아니다)

























