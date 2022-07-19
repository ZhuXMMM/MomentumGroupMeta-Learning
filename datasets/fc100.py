import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import numpy as np

from .datasets import register
from collections import Counter
import sys
np.set_printoptions(threshold=sys.maxsize)



@register('fc100')
class fc100(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train'
        split_file = 'FC100_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        # print('##################')
        # print(label)
        # print('##################')
        #对label按照顺序重新标号，例如[0 0 0 3 3 3 5 5 5]应标注为[0 0 0 1 1 1 2 2 2]
        label = torch.tensor(label)
        label, indices = torch.sort(label, dim=0)
        data = data[indices]
        label = label.tolist()
        ##################################################
        length = len(label)
        # label_loc = [i for i in range(length)]
        # new = np.vstack([np.array(label),np.array(label_loc)])
        # new = new.T[np.lexsort(new[::-1,:])].T
        if split == 'train':
            for i in range(length):
                num = i//600
                label[i] = num
                # new[0,i] = num
        elif split == 'test':
            for i in range(length):
                num = i//600
                label[i] = num
                # new[0,i] = 80 + num
        elif split == 'val':
            for i in range(length):
                num = i//600
                label[i] = num
                # new[0,i] = 60 + num
        # new = new.T[np.lexsort(new)].T
        # label = new[0,:]

        #################################################333
        # print('##################')
        # print(label)
        # print('##################')
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
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
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

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]

