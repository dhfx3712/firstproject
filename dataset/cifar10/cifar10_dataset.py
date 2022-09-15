import numpy as np
import os
import os
import numpy as np
from torch.utils.data import Dataset
import torch


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_traindata(file_path):
    train_data = None
    train_labels = None
    if not os.path.exists(file_path.format(1)):
        print('wrong dataset path : {}'.format(file_path.format(1)))
        exit()
    for i in range(5):#batch的5个文件
        data_dict = unpickle(file_path.format(i+1))
        if train_data is None:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.concatenate((train_data, data_dict[b'data']), axis=0)
            train_labels = np.concatenate((train_labels, data_dict[b'labels']), axis=0)
    # print(train_data.shape, train_labels.shape)
    return train_data, train_labels


class MyDataset(Dataset):
    def __init__(self, mode='train', root_path='D:\\data_cifia10\\'):
        super(MyDataset, self).__init__()
        if mode == 'train':
            file_path = os.path.join(root_path, 'data_batch_{}')
            print (f'file_path : {file_path} ,type : {type(file_path)}')
            self.data, self.labels = load_traindata(file_path=file_path)
        elif mode == 'test':
            file_path = os.path.join(root_path, 'test_batch')
            data_dict = unpickle(file_path)
            self.data = data_dict[b'data']
            self.labels = data_dict[b'labels']
        self.data = self.data/255
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index, :].reshape(3, 32, 32).astype(np.float32), self.labels[index]


train_loader = torch.utils.data.DataLoader(MyDataset('train', root_path='/Users/admin/data/cifar10/cifar-10-batches-py'), batch_size=128, shuffle=True)
print (f'train_loader : {len(train_loader)}') #128*391=50048 batch_size
test_loader = torch.utils.data.DataLoader(MyDataset('test', root_path='/Users/admin/data/cifar10/cifar-10-batches-py'))
print (f'test_loader : {len(test_loader)}')
# for index,data in enumerate(test_loader):
#     print (data[1])


'''
from sknet import SKNet

def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None):
    model.train()
    num = len(train_loader)
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
'''