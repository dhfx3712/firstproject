import pandas as pd
import shutil
import os



'''
df = pd.read_csv('/Users/admin/Downloads/trainLabels.csv')
df1 = df.groupby("label").sample(n=1000, random_state=1)

src = "/Users/admin/data/test_project/test1/"
des = "/Users/admin/data/test_project/test2/"


for index,items in df1.iterrows():
    try:
        shutil.copy(src+items["id"],des)
    except:
        print (items["id"])



df1 = pd.read_csv("/Users/admin/Downloads/label.csv")
print (df1.head())

dic = {}
for target in sorted(os.listdir("/Users/admin/Downloads/train1/")):
    dic[target] = 1

print (dic)

df1["tmp"] = df1["id"].map(dic)
print (df1.head())

df1 = df1[df1["tmp"]==1]

df1[["id","label"]].to_csv("/Users/admin/Downloads/111.csv",index=False)

'''

import os
import shutil

a = []
b = []
for i in os.listdir('/Users/admin/data/test_project/train_demo/test3/data/coco128/labels/train2017'):
    if not i.startswith("."):
        #print (i)
        a.append(i.split("#")[2].split('.')[0])
        os.rename('/Users/admin/data/test_project/train_demo/test3/data/coco128/labels/train2017/' +i,'/Users/admin/data/test_project/train_demo/test3/data/coco128/labels/train2017/' +i.split("#")[2])


for i in os.listdir('/Users/admin/data/test_project/train_demo/test3/data/coco128/images/train2017'):
    if not i.startswith("."):
        #print (i)
        b.append(i.split("#")[2].split('.')[0])
        os.rename('/Users/admin/data/test_project/train_demo/test3/data/coco128/images/train2017/' +i,'/Users/admin/data/test_project/train_demo/test3/data/coco128/images/train2017/' +i.split("#")[2])

print (len(set(a)&set(b)))