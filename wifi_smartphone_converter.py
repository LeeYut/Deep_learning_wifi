# -*- coding: utf-8 -*-    
import json
import pandas as pd
import numpy as np
#修改这里的文件名
file = "wifidataset 3 201801131322.csv"
df = pd.read_csv(file)
wifi_dict = {}
i=1
with open('test_file.txt', 'w') as file:
    for index, row in df.iterrows():
        if(row['BBSID'] == "delimiter"):
            if(125<=i<=225):
                print (wifi_dict)
                file.write(json.dumps(wifi_dict))
                wifi_dict = {}
			    #这里必须要有换行符，不然测试的时候出错
                file.write("\n")
            i += 1
        else:
            #if(int(row[' Signal'])>-80):
            wifi_dict[row['BBSID']] = row[' Signal']
	