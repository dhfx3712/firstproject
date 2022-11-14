import os
from sklearn.model_selection import train_test_split

tmp = []
label = []
for dir in os.listdir("/Users/admin/data/peopleface/face"):
    if os.path.isdir(os.path.join("/Users/admin/data/peopleface/face",dir)):
        if dir not in label:
            label.append(dir)
        for root, _ ,fnames in sorted(os.walk(os.path.join("/Users/admin/data/peopleface/face",dir))):
            for i in fnames:
                tmp.append([dir+'/'+i , dir])
print (label)



train,test = train_test_split(tmp,train_size=0.8,random_state=1)
print (len(tmp),len(train),len(test))

train_f = open('/Users/admin/data/peopleface/face/train.txt','w')
for i in train:
    train_f.write(i[0]+' '+i[1]+'\n')
train_f.close()


test_f = open('/Users/admin/data/peopleface/face/test.txt','w')
for i in test:
    test_f.write(i[0]+' '+i[1]+'\n')
test_f.close()


with open('/Users/admin/data/peopleface/face/label.txt','w') as f:
    for i in label:
        f.write(i+'\n')
