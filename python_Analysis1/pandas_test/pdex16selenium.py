# 셀레니움으로 임의의 사이트 화면 캡처

from selenium import webdriver

try:
    url = "http://www.daum.net"
    browser = webdriver.Chrome('C:/work/chromedriver')  # 내가 넣어준 경로명대로 작성해야함
    browser.implicitly_wait(3)
 
    browser.get(url);
    browser.save_screenshot("daum_img.png")
 
    browser.quit()
    print('성공')
except Exception:
    print('에러')