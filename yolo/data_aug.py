# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:07:47 2020

@author:
"""

import numpy as np
import cv2
import math


img1_path = '/Users/admin/data/test_project/coco128/images/train2017/000000000036.jpg'
img2_path = '/Users/admin/data/test_project/coco128/images/train2017/000000000049.jpg'




def mylearn():
    colors = [(0, 0, 255), (255, 0, 0)]#红色绘制原始框，蓝色绘制变换后的框
    lw = 1
    #voc数据集的一张图片数据
    img = cv2.imread('/Users/admin/data/test_project/coco128/images/train2017/000000000036.jpg')
    src = img.copy()
    h, w, c = img.shape
    cx, cy = w / 2, h / 2
    bboxs = np.loadtxt('/Users/admin/data/test_project/coco128/labels/train2017/000000000036.txt')
    cw, ch = 0.5 * bboxs[:, 3], 0.5 * bboxs[:, 4]
    bboxs[:, 3] = bboxs[:, 1] + cw
    bboxs[:, 4] = bboxs[:, 2] + ch
    bboxs[:, 1] -= cw
    bboxs[:, 2] -= ch
    bboxs[:, [1, 3]] *= w
    bboxs[:, [2, 4]] *= h
    srcboxs = bboxs.round().astype(np.int)
    #原始图像绘制bbox框
    for box in srcboxs:
        s = f'c{box[0]}'
        cv2.rectangle(src, (box[1], box[2]), (box[3], box[4]), color=colors[0], thickness=lw)
        cv2.putText(src, s, (box[1], box[2] - 2), cv2.FONT_HERSHEY_COMPLEX, 1.0, color=colors[0], thickness=lw)

    rotate = 10
    shear = 5
    scale = 0.8
    R, T1, T2, S, SH = np.eye(3), np.eye(3), np.eye(3), np.eye(3), np.eye(3)
    cos = math.cos(-rotate / 180 * np.pi)  # 图片坐标原点在左上角，该坐标系的逆时针与肉眼看照片方向相反
    sin = math.sin(-rotate / 180 * np.pi)
    R[0, 0] = R[1, 1] = cos  # 旋转矩阵
    R[0, 1] = -sin
    R[1, 0] = sin
    T1[0, 2] = -cx  # 平移矩阵
    T1[1, 2] = -cy
    T2[0, 2] = cx  # 平移矩阵
    T2[1, 2] = cy
    S[0, 0] = S[1, 1] = scale  # 缩放矩阵
    M = (T2 @ S @ R @ T1)  # 注意左乘顺序，对应，平移-》旋转-》缩放-》平移
    # M[:2]等价于cv2.getRotationMatrix2D(center=(cx, cy), angle=rotate, scale=scale)
    img = cv2.warpAffine(src, M[:2], (w, h), borderValue=(114, 114, 114))
    img=np.concatenate((src,img),axis=1)
    cv2.imwrite('affine.jpg', img)

    #再加上shear
    SH[0, 1] = SH[1, 0] = math.tan(shear / 180 * np.pi)  # 两个方向
    M = (T2 @ S @ SH @ T1)
    img = cv2.warpAffine(src, M[:2], (w, h), borderValue=(114, 114, 114))

    #bboxs坐标转换
    #srcboxs [n,5]
    # M矩阵用于列向量相乘，这里需要用转置处理所有坐标
    n=srcboxs.shape[0]
    xy = np.ones((n * 4, 3))#齐次坐标
    xy[:,:2]=srcboxs[:,[1,2,3,2,3,4,1,4]].reshape((n*4,2)) #顺时针xy,xy,xy,xy坐标
    transbox=(xy@M.T)[:,:2].reshape((n,8)).round().astype(np.int)
    for idx,box in enumerate(transbox):
        s = f'c{srcboxs[idx,0]}'
        cv2.line(img,(box[0], box[1]),(box[2], box[3]),color=colors[1],thickness=lw)
        cv2.line(img, (box[2], box[3]), (box[4], box[5]), color=colors[1], thickness=lw)
        cv2.line(img, (box[4], box[5]), (box[6], box[7]), color=colors[1], thickness=lw)
        cv2.line(img, (box[6], box[7]), (box[0], box[1]), color=colors[1], thickness=lw)
        cv2.putText(img, s, (box[0], box[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 1.0, color=colors[1], thickness=lw)
    img = np.concatenate((src, img), axis=1)
    cv2.imwrite('shrear.jpg',img)
    #透视变换
    #src=cv2.imread('../examples/test.png')
    P,RX,RY=np.eye(3),np.eye(3),np.eye(3)
    k=0.9
    def get_one_z(a,b,c):#z1,z2 get z (0~1)
        z1 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        z2 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        if z1>0 and z1<1:
            return z1
        else:
            return z2
    # 绕x轴旋转
    zx=get_one_z(1+w**2,-2,1-(k*w)**2)#一元二次方程求解
    #ax=math.atan((1-zx)/(w*zx))
    ax=math.asin((1-zx)/(k*w))
    cosx, sinx= math.cos(ax), math.sin(ax)
    RX[1, 1] = RX[2, 2] = cosx
    RX[1, 2] = -sinx
    RX[2, 1] = sinx
    img=cv2.warpPerspective(src,RX,(w,h))
    cv2.imwrite('perspective_rx.jpg', img)

    # 图像中心双轴旋转
    zy = get_one_z(1 + h ** 2, -2, 1 - (k * h) ** 2)  # 一元二次方程求解
    ay = math.atan((1 - zy) / (h * zy))
    cosy, siny = math.cos(ay), math.sin(ay)
    RY[0, 0] = RX[2, 2] = cosy
    RY[0, 2] = siny
    RY[2, 0] = -siny

    P=RX@RY
    print(P)
    M=T2@P@T1
    img = cv2.warpPerspective(src, M, (w, h))
    xy=xy @ M.T
    transbox = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8).round().astype(np.int)
    for idx, box in enumerate(transbox):
        s = f'c{srcboxs[idx, 0]}'
        cv2.line(img, (box[0], box[1]), (box[2], box[3]), color=colors[1], thickness=lw)
        cv2.line(img, (box[2], box[3]), (box[4], box[5]), color=colors[1], thickness=lw)
        cv2.line(img, (box[4], box[5]), (box[6], box[7]), color=colors[1], thickness=lw)
        cv2.line(img, (box[6], box[7]), (box[0], box[1]), color=colors[1], thickness=lw)
        cv2.putText(img, s, (box[0], box[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 1.0, color=colors[1], thickness=lw)
    img = np.concatenate((src, img), axis=1)
    cv2.imwrite('perspective.jpg',img)



# mylearn()



import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pylab

DEFAULT_PRNG = np.random


# 定义图片转换参数
class TransformParameters:
    def __init__(
            self,
            fill_mode='nearest',
            interpolation='linear',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':  # 最近邻插值
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':  # 双线性插值，适合放大图片
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':  # 4x4像素邻域的双三次插值，适合放大图片
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':  # 局部像素重采样，适合缩小图片
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4  # 8x8像素插值法


# 随机转动一定角度
def random_rotation(min, max, prng=DEFAULT_PRNG):
    angle = prng.uniform(min, max)

    rotation_reslut = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return rotation_reslut


# 随机平移
def random_translation(min, max, prng=DEFAULT_PRNG):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    translation = prng.uniform(min, max)
    translation_result = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])
    return translation_result


# 随机错切
def random_shear(min, max, prng=DEFAULT_PRNG):
    angle = prng.uniform()
    shear_result = np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])
    return shear_result


# 随机缩放
def random_scaling(min, max, prng=DEFAULT_PRNG):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    factor = prng.uniform(min, max)
    scaling_result = np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])
    return scaling_result


# 随机翻转
def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    factor = (1 - 2 * flip_x, 1 - 2 * flip_y)
    flip_result = np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])
    return flip_result


# 对图片进行随机变换，图片增强等操作，在不断的迭代训练中，其实是在变相增加训练集
def random_transform(
        min_rotation=0,
        max_rotation=0,
        min_translation=(0, 0),
        max_translation=(0, 0),
        min_shear=0,
        max_shear=0,
        min_scaling=(1, 1),
        max_scaling=(1, 1),
        flip_x_chance=0,
        flip_y_chance=0,
        prng=DEFAULT_PRNG):
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


# 创建图片变形生成器
def random_transform_generator(prng=None, **kwargs):
    if prng is None:
        prng = np.random.RandomState()
    while True:
        yield random_transform(prng=prng, **kwargs)


# 调整预处理
def adjust_transform_for_image(transform=None, image=None, relative_translation=True, transform_parameters=None):
    height, width, channels = image.shape
    result = transform  # tranform_generator
    if relative_translation:
        result[0:2, 2] *= [width, height]
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    image = apply_transform(transform, image, transform_parameters)
    return image


# 改变变形的中心点
def change_transform_origin(transform, center):
    center = np.array(center)

    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def translation(translation):
    return np.array([
        [1, 0, translation[0]],  # 0.5width, -0.5width
        [0, 1, translation[1]],  # 0.5height, -0.5height
        [0, 0, 1]
    ])


def apply_transform(matrix, image, params):
    output = cv2.warpAffine(
        image,  # 输入图像
        matrix[:2, :],  # 变换矩阵，为inputArray类型的3x3矩阵
        dsize=(image.shape[1], image.shape[0]),  # 输出图像的大小，尺寸保持不变
        flags=params.cvInterpolation(),  # 插值方法
        borderMode=params.cvBorderMode(),  # 边界像素模式
        borderValue=params.cval,  # 边界填充，默认值为0
    )
    return output



def test1():
    # 初始化一个图像处理参数对象
    transform_parameters = TransformParameters()
    # 创建一个图片转换迭代器
    # flip = {'flip_x_chance':0.5,'flip_y_chance':0.5}
    transform_generator = random_transform_generator()
    # 迭代
    for i, transform in enumerate(transform_generator):

        # 打开图片
        data_dir = '/Users/admin/data/test_project/coco128/images/train2017/'
        image_name = '000000000036'
        extension = '.jpg'
        # path = os.path.join(data_dir, image_name + extension)
        path = '/Users/admin/data/test_project/coco128/images/train2017/000000000036.jpg'
        image = cv2.imread('/Users/admin/data/test_project/coco128/images/train2017/000000000036.jpg')
        # 进行图片变形
        image_transformed = adjust_transform_for_image(transform=transform, image=image,
                                                       transform_parameters=transform_parameters)
        # 将图片转换成RGB图像
        image = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB)
        # 打印图片

        plt.imshow(image)
        plt.pause(3)


        # # 写入图片
        # path_write = os.path.join(data_dir, image_name + str(i) + extension)
        # cv2.imwrite(path_write, image_transformed)
        i += 1
        if i > 5:
            break




def get_batch(x, y, step, batch_size, alpha=0.2):
    """
    get batch data
    :param x: training data
    :param y: one-hot label
    :param step: step
    :param batch_size: batch size
    :param alpha: hyper-parameter α, default as 0.2
    :return:
    """
    candidates_data, candidates_label = x, y
    offset = (step * batch_size) % (candidates_data.shape[0] - batch_size)

    # get batch data
    train_features_batch = candidates_data[offset:(offset + batch_size)]
    train_labels_batch = candidates_label[offset:(offset + batch_size)]

    # 最原始的训练方式
    if alpha == 0:
        return train_features_batch, train_labels_batch
    # mixup增强后的训练方式
    if alpha > 0:
        weight = np.random.beta(alpha, alpha, batch_size)
        x_weight = weight.reshape(batch_size, 1, 1, 1)
        y_weight = weight.reshape(batch_size, 1)
        index = np.random.permutation(batch_size)
        x1, x2 = train_features_batch, train_features_batch[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = train_labels_batch, train_labels_batch[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return x, y

def mixup():
    img1 = cv2.imread(img1_path)
    img1 = cv2.resize(img1, (224, 224))

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (224, 224))

    for i in range(1, 10):
        lam = i * 0.1
        im_mixup = (img1 * lam + img2 * (1 - lam))
        plt.subplot(3, 3, i)
        plt.imshow(im_mixup/255)
    plt.show()



if __name__ == '__main__':
    # test1()
    mixup()

'''
过实际图像增强中不会去使用旋转、透视等，因为若这样做原始的box变换后是倾斜的不规则的，无法获取用于训练的外接矩形框

数据增强仿射变换
https://blog.csdn.net/ZHUYOUKANG/article/details/114481142

https://blog.csdn.net/weixin_41946992/article/details/105670935

数据增强其他方案
https://blog.csdn.net/u013685264/article/details/122622919

'''