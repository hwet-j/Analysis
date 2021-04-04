# GetOldTweets3  ===> 2021/04/04 막힌듯함.. 404 오류    

# 1. GetOldTweet3 패키지 설치 및 사용 준비

# pip install GetOldTweets3
import GetOldTweets3 as got  


# 2. 수집기간 정의
# 가져올 범위 정의 
import datetime # datetime 패키지 
days_range = [] 

# datetime 패키지, datetime 클래스(날짜와 시간 함께 저장) 
# strptime 메서드: 문자열 반환
start = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d") 
end = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d") 
date_generated = [start+datetime.timedelta(days=x) for x in range(0, (end-start).days)] 
for date in date_generated: 
    days_range.append(date.strftime("%Y-%m-%d")) 
     
print("===설정된 트윗 수집 기간: {} ~ {}===".format(days_range[0], days_range[-1])) 
print("===총 {}일 간의 데이터 수집===".format(len(days_range)))


# 3. 트윗 수집
# - 수집 기준 정의

# tweetCriteria로 수집 기준 정의
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('covid')\
                                           .setMaxTweets(1)
                                            #.setSince("2020-01-01")\
                                            #.setUntil("2020-12-31")\
print(1)
                                            
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
print(tweet.text)
import time

# 수집
print("데이터 수집 시작========")
start_time = time.time()

tweet = got.manager.TweetManager.getTweets(tweetCriteria)

print("데이터 수집 완료======== {0:0.2f}분".format((time.time() - start_time)/60))
print("=== 총 트윗 개수 {} ===".format(len(tweet)))

# 4. 변수 저장
# 원하는 정보만 저장 
# 유저 아이디, 트윗 내용, 날짜, 좋아요 수, 리트윗 수, 링크 

from tqdm.notebook import tqdm 
tweet_list = [] 
for i in tqdm(tweet): 
    username = i.username 
    tweet_date = i.date.strftime("%Y-%m-%d") 
    tweet_time = i.date.strftime("%H:%M:%dS") 
    content = i.text 
    favorites = i.favorites 
    retweets = i.retweets 
    link = i.permalink 
    
    info_list = [username, tweet_date, tweet_time, content, favorites, retweets, link] 
    tweet_list.append(info_list)

# 5. 파일 저장
import pandas as pd

# 데이터프레임 생성
twitter_df = pd.DataFrame(tweet_list, columns = ["ID", "날짜", "시간", "내용", "리트윗 수", "좋아요 수", "링크"])

# csv 파일 만들기
twitter_df.to_csv(("twitter_trump.csv"), index = False)
print("=== {}개의 트윗 저장 ===".format(len(tweet_list)))


# 6. 파일 확인
df_tweet = pd.read_csv("twitter_trump.csv".format(days_range[0], days_range[-1]))
df_tweet.head(10)


