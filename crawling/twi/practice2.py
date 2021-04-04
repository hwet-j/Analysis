import time
import datetime
import GetOldTweets3 as got
import logging
import logging.handlers
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import pandas as pd
import os

# 트윗 수집하는 함수 정의
# def get_tweets(start_date, end_date, keyword, keyword2):
def get_tweets(start_date, end_date, keyword):
    
    # 범위 끝을 포함하게 만듬
    end_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") 
                + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    # 트윗 수집 기준 설정
#     tweetCriteria = got.manager.TweetCriteria().setQuerySearch('{}'.format(keyword,keyword2))\
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('{}'.format(keyword))\
                                            .setSince(start_date)\
                                            .setUntil(end_date)\
                                            .setMaxTweets(-1) # 모두 수집
    print(got.manager.TweetManager)
    print("==> Collecting data start..")
    start_time = time.time()
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    print("==> Collecting data end.. {0:0.2f} minutes".format((time.time() - start_time)/60))
    print("=== Total number of tweets is {} ===".format(len(tweets)))
    
    return tweets
    
    
# 유저 리스트 반환하는 함수 정의
def get_users(tweets):
    
    user_list = []
    tweet_list = []

    for index in tweets:
        username = index.username
        content = index.text
        retweets = index.retweets
        favorites = index.favorites
        tweet_date = index.date.strftime("%Y-%m-%d")
        
        info_list = [tweet_date, username, content, retweets, favorites]
        tweet_list.append(info_list)
    
    return tweet_list
    
# logging 설정
def get_logger():
    logger = logging.getLogger("my")
    
    if len(logger.handlers) > 0:
        return logger
    
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    logger.addHandler(stream_hander)
    
    return logger
    
def crawl_userdata(username):
    
    # setting
    url = 'https://twitter.com/{}'.format(username)
#     mylogger.info("{} 유저의 데이터 수집 시작".format(username))
    HEADER = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    response = requests.get(url, headers=HEADER)
    html = response.text

    # parsing
    soup = BeautifulSoup(html, "lxml")

    # parsing fail
    try:
        user_profile_header = soup.find("div", {"class":'ProfileHeaderCard'})
        user_profile_canopy = soup.find("div", {"class":'ProfileCanopy-nav'})

        # data collect
        user = user_profile_header.find('a', {'class':'ProfileHeaderCard-nameLink u-textInheritColor js-nav'})['href'].strip("/") 

        date_joined = user_profile_header.find('div', {'class':"ProfileHeaderCard-joinDate"}).find('span', {'class':'ProfileHeaderCard-joinDateText js-tooltip u-dir'})['title']
        date_joined = date_joined.split("-")[1].strip()
        if date_joined is None:
            data_joined = "Unknown"

        tweets = user_profile_canopy.find('span', {'class':"ProfileNav-value"})['data-count']
        if tweets is None:
            tweets = 0
            
    except AttributeError:
#         mylogger.info("{} 유저의 데이터 수집 중 알수없는 오류가 발생했습니다.".format(username))
#         mylogger.info("링크 : {}".format(url))
        user, date_joined, tweets, following, followers = username, None, None, None, None
        
    # 블락 계정 특징 : 팔로워, 팔로잉 수가 안보임
    try:

        test_following = user_profile_canopy.find('li', {'class':"ProfileNav-item ProfileNav-item--following"})
        test_followers = user_profile_canopy.find('li', {'class':"ProfileNav-item ProfileNav-item--followers"})

        following = test_following.find('span', {'class':"ProfileNav-value"})['data-count']
        followers = test_followers.find('span', {'class':"ProfileNav-value"})['data-count']

#         mylogger.info("{} 유저의 데이터 수집 완료".format(username))

    except AttributeError:
#         mylogger.info("{} 유저는 블락된 계정입니다.".format(username))
        following = "Block"
        followers = "Block"
        
    os.system('clear')

    result = [user, date_joined, tweets, following, followers]
    
    return result
    
# 파일 저장
def save_file(tweet_list):
    twitter_df = pd.DataFrame(tweet_list, columns = ["tweet_date","username","content","retweets","favorites"])

    # csv 파일 생성
    twitter_df.to_csv("{}_{}_to_{}.csv".format(keyword, s_date, e_date), index=False)
    print("=== {} tweets are successfully saved ===".format(len(user_info)))

#     # 파일 확인
#     df_tweet = pd.read_csv('{}_{}_to_{}.csv'.format(keyword, start_date, end_date))
#     df_tweet.head(10) # 위에서 10개만 출력

# 유저 정보 Multiprocessing
global user_info
user_info = []

keyword = "from:3mindia"
# keyword2 = "samsung elec"
s_date = "2020-04-01"
e_date = "2020-04-30"

def main():
    # 유저 리스트 수집하기
#     tweets = get_tweets(s_date, e_date, keyword,keyword2)
    tweets = get_tweets(s_date, e_date, keyword)
    tweet_list = get_users(tweets)
    
#     user_list = users
    pool_size = len(tweet_list)
    
    if pool_size < 8:
        pool = Pool(pool_size)
        
    else:
        pool = Pool(8)
    
    for user in pool.map(crawl_userdata, tweet_list):
        user_info.append(user)
    
    save_file(tweet_list)
    
if __name__ == '__main__':
    
    start_time = time.time()
    
    mylogger = get_logger()
    mylogger.info("유저 정보 수집 시작")
    
    main()
    
    end_time = (time.time() - start_time)/60
    mylogger.info("유저 정보 수집 종료.. {0:0.2f} 분 소요".format(end_time))
    mylogger.info("총 수집된 유저 정보는 {} 개 입니다.".format(len(user_info)))
    