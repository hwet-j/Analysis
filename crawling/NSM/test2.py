# 긍정 1 부정 -1 그외 0

with open("./negative_words_twit.txt", encoding='utf-8') as neg: 
    negative = neg.readlines()

negative = [neg.replace("\n", "") for neg in negative] 

with open("./positive_words_twit.txt", encoding='utf-8') as pos: 
    positive = pos.readlines() 
    
negative = [neg.replace("\n", "") for neg in negative] 
positive = [pos.replace("\n", "") for pos in positive]


from tqdm import tqdm 
import re 
import pandas as pd

my_title_df = pd.read_csv("Twitter_all_data.txt", sep="\n", header=None)

# my_title_df = pd.DataFrame(my_title_df, columns="title")

print(my_title_df.head(4))
print(my_title_df.columns)

my_title_df = my_title_df.rename(columns={0:"title"})

print(my_title_df.columns)

labels = [] 


title_data = list(my_title_df['title']) 

for title in tqdm(title_data): 
    clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title) 
    negative_flag = False 
    label = 0 
    for i in range(len(negative)): 
        if negative[i] in clean_title: 
            label = -1 
            negative_flag = True 
            print("negative 비교단어 : ", negative[i], "clean_title : ", clean_title) 
            break 
    if negative_flag == False: 
        for i in range(len(positive)): 
            if positive[i] in clean_title: 
                label = 1 
                print("positive 비교단어 : ", positive[i], "clean_title : ", clean_title) 
                break 
        
    labels.append(label) 
    
my_title_df['label'] = labels
   
print(my_title_df)

print(my_title_df['label'].value_counts())

my_title_df.to_excel(excel_writer="result.xlsx")
