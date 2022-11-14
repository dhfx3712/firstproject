import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt


def mv_file():
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
from PIL import Image
import numpy as np
def load_pic():
    tif = Image.open('/Users/admin/Downloads/cell/masks/21.tif')
    print (f'tif_shape :{np.shape(tif)},{np.asarray(tif)[230,300:400]}')

    gif = Image.open('/Users/admin/Downloads/car/masks/0cdf5b5d0ce1_14_mask.gif')
    print(f'gif_shape :{np.shape(gif)},{np.asarray(gif)[637, 700:800]}')

    # voc = Image.open('/Users/admin/data/VOCdevkit/VOC2007/SegmentationClass/000129.png')
    voc = Image.open('/Users/admin/data/VOCdevkit/VOC2007/JPEGImages/000129.jpg')
    print (f'voc_shape :{np.shape(voc)},{np.asarray(voc)[337, 200:300]}')
    plt_img(voc,"orgin")
    voc_array = np.asarray(voc)
    voc_array[voc_array==0]=200
    print (f'255-0 : {np.shape(voc_array)},{voc_array[337, 200:300]}')
    voc1 = Image.fromarray(voc_array)
    plt_img(voc1,"255-0")
    voc2 = voc1.resize((200,200),resample=Image.NEAREST)
    plt_img(voc2, "resize")


'''
二值图片（0黑，1白）
灰度（0-255之间）
彩色（RGB三通道，每个通道0-255，每个通道均是一个灰度图片）
'''


def plt_img( img, title):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_title(title)
    ax.imshow(img)
    plt.show()

if __name__=="__main__":
    # mv_file()
    load_pic()