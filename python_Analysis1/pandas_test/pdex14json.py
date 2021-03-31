# json 처리
import json

json_file = "./pdex14.json"
json_data = {}

def readData(filename):
    f = open(filename, 'r', encoding="utf-8")
    lines = f.read()
    f.close()
    #print(lines)
    return json.loads(lines)    # decoding str --> dict
    
def main():
    global json_file
    json_data = readData(json_file)
    print(json_data)  # <class 'dict'>
    #print(type(json_data))  # <class 'dict'>
    
    d1 = json_data['직원']['이름']  # 직원 하위의 '이름'에 대응하는 value값
    d2 = json_data['직원']['직급']
    d3 = json_data['직원']['전화']
    print("이름 : " + d1 + ",직급 : " + d2 + ",전화 : " + d3)
    
    
if __name__=="__main__":
    main()















