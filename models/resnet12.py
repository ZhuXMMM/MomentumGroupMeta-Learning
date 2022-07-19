import torch.nn as nn
import torch
from .models import register
import torch.nn.functional as F

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out
# todo Bottleneck
class Bottleneck(nn.Module):
    """
    __init__
        in_channel：残差块输入通道数
        out_channel：残差块输出通道数
        stride：卷积步长
        downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
    """
    expansion = 4   # 残差块第3个卷积层的通道膨胀倍率
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  # H/2，W/2。C不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x    # 将原始输入暂存为shortcut的输出
        if self.downsample is not None:
            identity = self.downsample(x)   # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity     # 残差连接
        out = self.relu(out)

        return out


class ResNet12(nn.Module):
    expansion = 4
    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x) #128,3
        x = self.layer2(x) #128,64
        x = self.layer3(x) #128,128
        x = self.layer4(x) #128,256->128,512
        # x = torch.flatten(x, 1)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x

#resnet50
# todo ResNet
class ResNet50(nn.Module):
    """
    __init__
        block: 堆叠的基本模块
        block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
        num_classes: 全连接之后的分类特征维度
        
    _make_layer
        block: 堆叠的基本模块
        channel: 每个stage中堆叠模块的第一个卷积的卷积核个数，对resnet50分别是:64,128,256,512
        block_num: 当期stage堆叠block个数
        stride: 默认卷积步长
    """
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channel = 64    # conv1的输出维度

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   # H,W不变。downsample控制的shortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048
        self.out_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

        for m in self.modules():    # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None   # 用于控制shorcut路的
        if stride != 1 or self.in_channel != channel*4:   # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4，因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                nn.BatchNorm2d(num_features=channel*block.expansion))

        layers = []  # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
        self.in_channel = channel*block.expansion   # 在下一次调用_make_layer函数的时候，self.in_channel已经x4

        for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)#128，2048
        # x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x




@register('resnet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])


@register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])

def resnet50(num_classes=512):
    return ResNet50(block=Block, block_num=[3, 4, 6, 3], num_classes=num_classes)

# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from torch.distributions import Bernoulli

# # This ResNet network was designed following the practice of the following papers:
# # TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# # A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# class DropBlock(nn.Module):
#     def __init__(self, block_size):
#         super(DropBlock, self).__init__()

#         self.block_size = block_size
#         #self.gamma = gamma
#         #self.bernouli = Bernoulli(gamma)

#     def forward(self, x, gamma):
#         # shape: (bsize, channels, height, width)

#         if self.training:
#             batch_size, channels, height, width = x.shape
            
#             bernoulli = Bernoulli(gamma)
#             mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
#             #print((x.sample[-2], x.sample[-1]))
#             block_mask = self._compute_block_mask(mask)
#             #print (block_mask.size())
#             #print (x.size())
#             countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
#             count_ones = block_mask.sum()

#             return block_mask * x * (countM / count_ones)
#         else:
#             return x

#     def _compute_block_mask(self, mask):
#         left_padding = int((self.block_size-1) / 2)
#         right_padding = int(self.block_size / 2)
        
#         batch_size, channels, height, width = mask.shape
#         #print ("mask", mask[0][0])
#         non_zero_idxs = mask.nonzero()
#         nr_blocks = non_zero_idxs.shape[0]

#         offsets = torch.stack(
#             [
#                 torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
#                 torch.arange(self.block_size).repeat(self.block_size), #- left_padding
#             ]
#         ).t().cuda()
#         offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
#         if nr_blocks > 0:
#             non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
#             offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
#             offsets = offsets.long()

#             block_idxs = non_zero_idxs + offsets
#             #block_idxs += left_padding
#             padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
#             padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
#         else:
#             padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
#         block_mask = 1 - padded_mask#[:height, :width]
#         return block_mask


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.LeakyReLU(0.1)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv3x3(planes, planes)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.maxpool = nn.MaxPool2d(stride)
#         self.downsample = downsample
#         self.stride = stride
#         self.drop_rate = drop_rate
#         self.num_batches_tracked = 0
#         self.drop_block = drop_block
#         self.block_size = block_size
#         self.DropBlock = DropBlock(block_size=self.block_size)
        

#     def forward(self, x):
#         self.num_batches_tracked += 1

#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         out = self.maxpool(out)
        
#         if self.drop_rate > 0:
#             if self.drop_block == True:
#                 feat_size = out.size()[2]
#                 keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
#                 gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
#                 out = self.DropBlock(out, gamma=gamma)
#             else:
#                 out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

#         return out


# class ResNet(nn.Module):

#     def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0, dropblock_size=5):
#         self.inplanes = 3
#         super(ResNet, self).__init__()

#         self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
#         self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
#         self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
#         self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
#         self.out_dim = 640
#         if avg_pool:
#             self.avgpool = nn.AvgPool2d(5, stride=1)

#         self.keep_prob = keep_prob
#         self.keep_avg_pool = avg_pool
#         self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
#         self.drop_rate = drop_rate
#         #self.emb_size = 160

#         self.downsample_layer = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
#                                             nn.BatchNorm2d(160)
#                                             )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
#         self.inplanes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         #print(x.size())
#         #x = self.downsample_layer(x)

#         if self.keep_avg_pool:
#             x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return x

# @register('resnet12')
# def resnet12(keep_prob=1.0, avg_pool=True, **kwargs):
#     """Constructs a ResNet-12 model.
#     """
#     model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
#     return model