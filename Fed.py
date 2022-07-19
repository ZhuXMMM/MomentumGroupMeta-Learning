#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import copy
import random


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def cal_weight(weight_old,distance,num):
    weight=[]
    n=0
    for i in distance:
        weight.append(float(i/(sum(distance))))
        n=n+1
    return weight

def weight_norm(acc):
    acc_norm = []
    for i in acc:
        acc_norm.append(i/sum(acc))
    return acc_norm

def a_cal(train_batch_size = 50, n=0):
    a=(0.2+0.5*n/train_batch_size)
    return a

def FedCDW(w, weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():       
        w_avg[k] = torch.tensor(weight)*w[0][k]
        for i in range(1, len(w)):
            w_avg[k] += torch.tensor(1-weight)*w[i][k]
    return w_avg

def FedSTS(w, weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        num=random.random()
        if num >= 0.5:
            w_avg[k] = torch.tensor(weight)*w[0][k]
            w_avg[k] += torch.tensor(1-weight)*w[1][k]
        else :
            w_avg[k] = w[0][k]
    return w_avg

def Fedclient(w, weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.tensor(float(weight[0]))*w[0][k]
        for i in range(1, len(w)):
            w_avg[k] += torch.tensor(float(weight[i]))*w[i][k]
    return w_avg

def softmax_weight(weight):
    m = nn.Softmax()
    client_weight = m(torch.tensor(weight))
    return client_weight