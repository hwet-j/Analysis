# 필요없는 파일(테스트용)
from twit_crawl.Craw import twitter_search
from twit_crawl.Clean import divide_word
from twit_crawl.WordCloud import make_wc

# django > views 에서 검색해서 사용
if __name__ == '__main__':
    twitter_search("(아스트라제네카 OR AZ백신)")
    divide_word("(아스트라제네카 OR AZ백신)")
    #make_wc("AZ백신")
    