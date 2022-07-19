import argparse
import os
import yaml
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Fed import FedAvg,cal_weight,softmax_weight,Fedclient,FedCDW
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from mycrossentropy import CELoss,MYCELoss
import collections as cl
import torch.nn as nn
m=nn.Sigmoid()

def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    if float(args.momentum)==0.0: #GML
        save_path = os.path.join('./save', svname,"double_GPU_new","_batch_size_"+args.batch_size+"_user_num_"+args.task_nums) 
    elif float(args.momentum)==1.0:#AMS
        save_path = os.path.join('./save', svname,"double_GPU_new","_batch_size_"+args.batch_size+"_user_num_"+args.task_nums+'_momentum_b05') 
    else:#MS
        save_path = os.path.join('./save', svname,"double_GPU_new","_batch_size_"+args.batch_size+"_user_num_"+args.task_nums+'_momentum_'+str(float(args.momentum)))
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
        task_nums = int(args.task_nums)
        batch_size=int(args.batch_size)
        ep_per_batch_train = batch_size*task_nums
        train_batch_size = int(int(args.task)/ep_per_batch_train)
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, train_batch_size,
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch_train)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
                tval_dataset.label, int(800/batch_size),
                n_way, n_shot + n_query,
                ep_per_batch=batch_size)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, int(800/batch_size),
            n_way, n_shot + n_query,
            ep_per_batch=batch_size)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    #####################################

    ######## Model and optimizer ########
    ##############global#################
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)    
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])


    ##############group#################
    net = models.make(config['model'], **config['model_args'])
    if config.get('load_encoder'):
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        net.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        net = nn.DataParallel(net)
    net_optimizer, lr_scheduler = utils.make_optimizer(
            net.parameters(),
            config['optimizer'], **config['optimizer_args'])
    #####################################
    
    max_epoch = int(args.max_epoch)
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    train_acc=[]
    tval_acc=[]
    if float(args.momentum)==1.0:
        momentum = 0.8
        momentum_last = 0.8
    else:
        momentum = float(args.momentum)
        print('momentum = ',momentum)
    loss_store = cl.deque(maxlen=3)
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        w_Momentum = cl.deque(maxlen=2)
        w_Momentum.append(copy.deepcopy(model.state_dict())) 
        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            w_locals=[]
            loss_locals=[]
            acc_locals=[]
            for index in range(int(task_nums)):
                x_shot, x_query = fs.split_shot_query(
                        data[index*batch_size*(75+5*n_train_shot):(index+1)*batch_size*(75+5*n_train_shot)].cuda(), n_train_way, n_train_shot, n_query,
                        ep_per_batch=batch_size)

                label = fs.make_nk_label(n_train_way, n_query,
                        ep_per_batch=batch_size).cuda()
                #Copy the network parameters of the global model
                net.load_state_dict(copy.deepcopy(model.state_dict()))
                task_epoch = int(args.task_epoch)
                #Each meta learning task trains task_epoch times
                for i in range(0,task_epoch):
                    logits = net(x_shot, x_query).view(-1, n_train_way)
                    net_loss = F.cross_entropy(logits, label)
                    net_acc = utils.compute_acc(logits, label)
                    net_optimizer.zero_grad()
                    net_loss.backward()
                    net_optimizer.step()
                    if i == task_epoch-1:
                        w_locals.append(copy.deepcopy(net.state_dict()))
                        loss_locals.append(net_loss)
                        acc_locals.append(copy.deepcopy(net_acc))
                    logits = None; net_loss = 0
            #Model aggregation
            w_glob = FedAvg(w_locals)
            #######AMS########
            w_Momentum.append(copy.deepcopy(w_glob)) 
            #Calculates the global features of the next stage, and the formula is$w_{i}^{\ast} = \alpha*w_{i-1}+(1-\alpha)*w_{i}$，
            w_new = FedCDW(w_Momentum, momentum)

            w_Momentum.append(copy.deepcopy(w_new))
            loss = sum(loss_locals) / len(loss_locals)
            acc = sum(acc_locals) / len(acc_locals)
            model.load_state_dict(w_new)
            aves['tl'].add(loss.item())
            aves['ta'].add(acc)
        # eval
        model.eval()

        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=batch_size)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=batch_size).cuda()

                with torch.no_grad():
                    logits = model(x_shot, x_query).view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, momentum {:.4f} {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'],momentum, t_epoch, t_used, t_estimate, _sig))

        train_acc.append(copy.deepcopy(aves['ta']))
        tval_acc.append(copy.deepcopy(aves['tva']))
        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va-'+str(epoch)+'.pth'))

        

        writer.flush()
        if float(args.momentum)==1.0:
            if epoch/17 >=1:
                epoch_rate = 1
            else:
                epoch_rate = epoch/17
            loss_store.append(copy.deepcopy(aves['tl']))
            if len(loss_store) == 3:
                diff = (loss_store[2]-loss_store[1])/(loss_store[1]-loss_store[0])-1
                momentum = float(1-m(torch.tensor(diff)))
                if momentum > args.upper:
                    momentum = args.upper
                else:
                    momentum = float(1-m(torch.tensor(diff)))*epoch_rate
            momentum_last = momentum
    print('train_acc:',train_acc)
    print('tval_acc:',tval_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default = 'configs/train_meta_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='1,0')
    parser.add_argument('--task_nums', default='5')
    parser.add_argument('--batch_size', default='4')
    parser.add_argument('--task_epoch', default='1')
    parser.add_argument('--max_epoch', default='20')
    parser.add_argument('--task', default='800')
    parser.add_argument('--momentum',type = float,default=1.0)
    parser.add_argument('--upper',type = float,default=1.0)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

