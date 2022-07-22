import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class SPP_NET(nn.Module):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''

    def __init__(self, opt, input_nc, ndf=64, gpu_ids=[]):
        super(SPP_NET, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = [4, 2, 1]

        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(10752, 4096)
        self.fc2 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LReLU1(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.BN2(x))

        x = self.conv4(x)
        # x = F.leaky_relu(self.BN3(x))
        # x = self.conv5(x)
        spp = spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num)
        # print(spp.size())
        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)
        s = nn.Sigmoid()
        output = s(fc2)
        return output


def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


# model = SPP_NET()



# x  = tf.ones((4,16,16,3))
# x_sppl = spp_layer(x,4)

'''


# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

def spp_layer(input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):


    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name):

        for l in range(levels):
        #设置池化参数.[1,2,3,4]尺度
            l = l + 1
            ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
            
            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1),)
                
            else :
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1：
                x_flatten = tf.reshape(pool,(shape[0],-1))
            else:
                x_flatten = tf.concat((x_flatten,pool),axis=1) #四种尺度进行拼接
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
            

    return x_flatten


#x  = tf.ones((4,16,16,3))
#x_sppl = spp_layer(x,4)





'''