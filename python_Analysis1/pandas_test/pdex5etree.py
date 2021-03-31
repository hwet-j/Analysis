# XML, HTML 문서 읽기 

import xml.etree.ElementTree as etree

# 일반적인 형태의 파일 읽기
xml_f = open("pdex5.xml", "r", encoding="utf-8").read()
print(xml_f, '\n', type(xml_f)) # <class 'str'> 이므로 str 관련 명령만 사용가능

print()
root = etree.fromstring(xml_f)
print(root, '\n', type(root))  # <class 'xml.etree.ElementTree.Element'>  Element 관련 명령 사용가능
print(root.tag, len(root.tag))

print('---------' * 3)
xmlfile = etree.parse("pdex5.xml")
print(xmlfile)
root = xmlfile.getroot()
print(root.tag)  # 태그값(명)을 가져옴
print(root[0].tag)
print(root[0][0].tag)
print(root[0][2].tag)
print(root[0][2].attrib)    # 속성값을 가져옴
print(root[1][2].attrib.keys())    # 속성값의 key만 가져옴
print(root[1][2].attrib.values())    # 속성값의 value만 가져옴



