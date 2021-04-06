import pandas as pd
from datetime import datetime, timedelta
for i in range(5):
    end = (datetime.today() - timedelta(i)).strftime("%Y-%m-%d")
    print("tweet_{}.txt".format(end))
