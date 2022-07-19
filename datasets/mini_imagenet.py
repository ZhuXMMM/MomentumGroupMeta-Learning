import os
import pickle
from PIL import Image
from cv2 import transpose

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from .datasets import register
import numpy as np
np.set_printoptions(threshold=np.inf)
from collections import Counter
# from .datasets import datastrong.AddPepperNoise

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """
#1
    def __init__(self, snr, p=1.0):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p
#2
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:#概率的判断
            img_ = np.array(img).copy()#转化为numpy的形式
            h, w ,c  = img_.shape#获取图像的高，宽，channel的数量
            signal_pct = self.snr#设置图像原像素点保存的百分比
            noise_pct = (1 - self.snr)#噪声的百分比
            mask = np.random.choice((0, 1, 2), size=( h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            #random.choice的用法：可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
            #size表示要输出的numpy的形状
            mask = np.repeat(mask, c, axis=2)#将mask在最高轴上（2轴）上复制channel次。
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            # img_ = img_.transpose(1,2,0)
            return Image.fromarray(img_.astype('uint8')).convert('RGB')#转化成pil_img的形式
        else:
            return img


@register('mini-imagenet')
class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        # print('##################')
        # print(Counter(label))
        # print('##################')
        image_size = 80
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        #正常
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
        #椒盐噪声
        self.PepperNoise = transforms.Compose([
            transforms.Resize(image_size),
            AddPepperNoise(snr = 0.90),
            transforms.ToTensor(),
            normalize,  
        ])
        #高斯模糊
        self.GaussNoise = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.GaussianBlur(7, 5),
            normalize,  
        ])
        #视角变形
        self.persperctive = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=1),
            normalize,  
        ])
        #随机擦除
        self.erasing = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.2,0.33), ratio=(0.33, 3.0), value=(0,0,0)),
            normalize,  
        ])



        augment = kwargs.get('augment')#None
        # # print(augment)
        # print('##################')
        # print(augment)
        # print('##################')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform
            self.transform_PepperNoise = self.PepperNoise
            self.transform_GaussNoise = self.GaussNoise
            self.transform_persperctive = self.persperctive
            self.transform_erasing = self.erasing


        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        mode = 'bad'
        if mode == 'bad':
            p = random.random()
            if p>=0.0:
                return self.transform(self.data[i]), self.label[i]
            else:
                q = random.random()
                if q>0 and q<=0.25:
                    return self.transform_PepperNoise(self.data[i]), self.label[i]
                elif q>0.25 and q<=0.5:
                    return self.transform_GaussNoise(self.data[i]), self.label[i]
                elif q>0.5 and q<=0.75:
                    return self.transform_persperctive(self.data[i]), self.label[i]
                    # return self.transform_GaussNoise(self.data[i]), self.label[i]
                elif q>0.75 and q<=1.0:
                    return self.transform_erasing(self.data[i]), self.label[i]
        elif mode == 'normal':
            return self.transform_persperctive(self.data[i]), self.label[i]


