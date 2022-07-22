import pandas as pd
import os
import shutil

src = "/Users/admin/data/test_project/train_demo/test3/data/coco128/images/train2017/"
des = "/Users/admin/project/tmp/person"
for i in os.listdir('/Users/admin/data/test_project/train_demo/test3/workspace/result/'):
    if not i.startswith("."):
        #print (i)
        name = i.split(".")[0]
        try:
            shutil.copy(src + name + ".jpg", des)
        except:
            print(name)


'''
df = pd.read_csv("/Users/admin/data/peopel.csv",header=None,sep=" ")
print (df.head())
df[5] = df[0].str.split(":",expand=True)[0]
df[0] = df[0].str.split(":",expand=True)[1]
print (df.head())
'''





'''
data = pd.read_csv('./train.txt',sep='\t',names=['path','label'])
print (data.head())
for index,items in data.iterrows():
    shutil.copy('../%s'%items["path"], './')
~                                              
'''

