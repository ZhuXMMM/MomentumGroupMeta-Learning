import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('encoder-baseline')
class EncoderBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, data):
        x_tot = self.encoder(data)
        x_tot = x_tot.unsqueeze(0)
        x_tot = x_tot.mean(dim=1)
        x_tot = F.normalize(x_tot, dim=-1)
        return(x_tot)

