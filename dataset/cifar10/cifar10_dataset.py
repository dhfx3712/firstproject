from sknet import SKNet



train_loader = torch.utils.data.DataLoader(MyDataset('train', root_path=root_path), batch_size=128, shuffle=True)


class MyDataset(Dataset):
    def __init__(self, mode='train', root_path='D:\\data_cifia10\\'):
        super(MyDataset, self).__init__()
        if mode == 'train':
            file_path = os.path.join(root_path, 'data_batch_{}')
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

def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None):
    model.train()
    num = len(train_loader)
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
