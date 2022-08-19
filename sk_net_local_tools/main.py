import os
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
from dataset import MyDataset
from resnext import ResNeXt
from sknet import SKNet
from train import train_epoch, test


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__=="__main__":

    root_path = '../../cifar10/cifar-10-batches-py'

    train_loader = torch.utils.data.DataLoader(MyDataset('train', root_path=root_path), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MyDataset('test', root_path=root_path), batch_size=16)

    net = ResNeXt(10)
    # net = SKNet(10) #因为inceptionv3报错的地方kernel大小是8，如果图像输入过小的话，到这一层的时候输入就会很小，无法进行相应的操作
    # net.cuda()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, betas=(0.9, 0.999))
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()

    log_path = './logs/'
    writer = SummaryWriter(log_path)

    epoch_num = 3
    lr0 = 1e-3
    for epoch in range(epoch_num):
        current_lr = lr0 / 2**int(epoch/50)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        train_epoch(net, optimizer, train_loader, criterion, epoch, writer=writer)
        test(net, test_loader, criterion, epoch, writer=writer)
        if (epoch+1)%5==0:
            torch.save(net.state_dict(), os.path.join('./model/model_{}.pth'.format(epoch)))
    torch.save(net.state_dict(), os.path.join('./model/model.pth'))