# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch


# train_datasets = ConeDateSet( args.train_dir, transforms=None)


class ConeDateSet(data.Dataset):

    def __init__(self, database ,lists, transforms=None, train=True, test=False):

        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]

        with open(lists, 'r') as f:
            lines = f.readlines()

        imgs = []
        labels = [] #原始label信息
        label_uniq = set() #类别总数
        print ("lines:%s"%len(lines))

        for line in lines:
            # print(line)
            # a = os.path.join(root, line.split()[0])
            if os.path.exists(os.path.join(database,line.split()[0])):
                imgs.append(os.path.join(database,line.split()[0]))
                # b = int(line.split()[1])
                labels.append(int(line.split()[1]))
                if int(line.split()[1]) not in label_uniq:
                    label_uniq.add(int(line.split()[1]))

        self.imgs = imgs
        # print (self.imgs)
        self.labels = labels
        print (label_uniq)

        label_uniq = list(label_uniq)
        label_uniq.sort()
        print (label_uniq)

        self.label2ix ={data:index for index,data in enumerate(label_uniq)} #排序编码后的label信息
        print (self.label2ix)

        if transforms is None:

            self.transforms = T.Compose([
                T.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.1),
                # T.Grayscale(num_output_channels=1),
                T.Resize((380, 160)),
                # T.FiveCrop(100),
                # T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                # T.Lambda(lambda tensors: torch.stack([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #                                      (t) for t in tensors])),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.RandomRotation(45),
                # T.CenterCrop(100),
                # T.RandomCrop(100),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert("RGB")
        # data = np.array(data)
        # data = data.astype(np.uint8)
        # data = Image.fromarray(data, 'P')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':

    dataset = ConeDateSet( '/Users/admin/data/cls','/Users/admin/data/cls/train_helmet.txt')
    # img, label = dataset[0]
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)


'''
return data,label

dataset

(base) admin@bogon cls % tree -L 1
.
├── helmet
├── nohelmet
├── train_helmet.txt
└── val_helmet.txt


图片数据在helmet和nohelmet
训练数据和验证数据在*.txt
helmet/02594300_0214_1_0.9967256784439087.jpg	0


'''