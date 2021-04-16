import pandas as pd

# with open("./data/freq_AZ.txt", encoding='utf-8') as text: 
#     for line in text:
#         data = line.line().rstrip("\n")
#     
data = pd.read_csv("./data/train_d.csv" ,names=["x_train", "y_train"])

print(data[:30])