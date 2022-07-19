import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

class CategoriesSampler():
#label,共有n_batch*ep_per_batch个task,n_cls=5，n_per=1+15
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                #随机挑选5个类
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    #挑选出的5个类每个类选出1+15张图
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)

