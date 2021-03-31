# 공공데이터(외국인 관광객의 국내 관광지 입장자료)로 상관관계 분석
import json
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import pandas as pd

def Start():
    # 서울시 관광지 정보
    fname = "서울_관광지.json"
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    # print(jsonTP)
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm','resNm','ForNum'))  # 년월, 관광지명, 입장객수
    tour_table = tour_table.set_index('yyyymm')
    # print(tour_table)
    resNm = tour_table.resNm.unique()
    print('resNm : ', resNm[:5])    # ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']
    
    # 중국인 관광정보
    cdf = '중국인방문객.json'
    cdata = json.loads(open(cdf, 'r', encoding='utf-8').read())
    # print(cdata)
    china_table = pd.DataFrame(cdata, columns=('yyyymm','visit_cnt'))
    china_table = china_table.rename(columns = {'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    # print(china_table)
    
    # 일본인 관광정보
    jdf = '일본인방문객.json'
    jdata = json.loads(open(jdf, 'r', encoding='utf-8').read())
    # print(jdata)
    japan_table = pd.DataFrame(jdata, columns=('yyyymm','visit_cnt'))
    japan_table = japan_table.rename(columns = {'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    # print(china_table)
    
    # 미국인 관광정보
    udf = '일본인방문객.json'
    udata = json.loads(open(udf, 'r', encoding='utf-8').read())
    # print(udata)
    usa_table = pd.DataFrame(udata, columns=('yyyymm','visit_cnt'))
    usa_table = usa_table.rename(columns = {'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    # print(china_table)
    
    # index를 기준으로 left조인 (left_index=True)
    all_table = pd.merge(china_table, japan_table, left_index = True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index = True, right_index=True)
    # print(all_table)
    
    r_list = []
    
    for tourPoint in resNm[:5]:
        r_list.append(SetScrapperGraph(tour_table, all_table, tourPoint))
        # print(r_list)
        
    r_df = pd.DataFrame(r_list, columns=('관광지명','중국','일본','미국'))
    r_df = r_df.set_index('관광지명')
    print(r_df)
    
    r_df.plot(kind='bar', rot=60)
    plt.show()
        
        
def SetScrapperGraph(tour_table, all_table, tourPoint):
    tour = tour_table[tour_table['resNm'] == tourPoint]     
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index = True, right_index=True)
    # print(merge_table)  # 관광지 자료 중 앞에 5개만 참여
    
    # 시각화 + 상관관계
    fig = plt.figure()
    fig.suptitle(tourPoint + '상관관계분석')
    # 중국
    plt.subplot(1, 3, 1)
    plt.xlabel('중국인 수')
    plt.ylabel('외국인 입장수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    # print('r1 : ', r1)
    plt.title('r={:.3f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], s = 6, c = 'black')
    
    # 일본
    plt.subplot(1, 3, 2)
    plt.xlabel('중국인 수')
    plt.ylabel('외국인 입장수')
    lamb1 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb1(merge_table)
    # print('r2 : ', r2)
    plt.title('r={:.3f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], s = 6, c = 'red')
    
    # 미국
    plt.subplot(1, 3, 3)
    plt.xlabel('중국인 수')
    plt.ylabel('외국인 입장수')
    lamb1 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb1(merge_table)
    # print('r1 : ', r1)
    plt.title('r={:.3f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], s = 6, c = 'blue')
    
    # plt.show()
    
    return [tourPoint, r1, r2, r3]
    
    
    
if __name__ == '__main__':
    Start()























