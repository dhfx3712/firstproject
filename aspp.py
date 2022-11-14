from torch import nn
import torch
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel):
        depth = in_channel
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        print (f'mean : {image_features.shape},{image_features[0,0,:]}')
        image_features = self.conv(image_features)
        print (f'conv : {image_features.shape},{image_features[0,0,:]}')
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
        print (f'interplote : {image_features.shape},{image_features[0,0,0,:]}')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        print(f'atrous1 : {atrous_block1.shape} ,atrous6 : {atrous_block6.shape} ,atrous12 : {atrous_block12.shape} ,,atrous18 : {atrous_block18.shape}')

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net


aspp = ASPP(256)
out = torch.rand(2, 256, 13, 13)
print(aspp(out).shape)

'''
还是每个空洞卷积的输出，大小都是一样的，所以在通道维度可以concat
2p-k-(kd-k-d+1)=2p-k-kd+k+d-1=-1 输入和输出一致
https://blog.csdn.net/weixin_42475184/article/details/115394357
'''