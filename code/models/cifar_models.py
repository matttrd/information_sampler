'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feats = out
        out = self.linear(out)
        return out, feats


class wideresnet(nn.Module):
    name = 'wideresnet'

    def __init__(self, opt, num_classes):
        super(wideresnet, self).__init__()

        opt['depth'] = opt.get('depth', 28)
        opt['widen'] = opt.get('widen', 10)

        d, depth, widen = 0., opt['depth'], opt['widen']

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)//6

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def block(ci, co, s, p=0.):
            h = nn.Sequential(
                    bn2(ci,track_running_stats=track_running_stats),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, bias=False),
                    bn2(co,track_running_stats=track_running_stats),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p),
                    nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False))
            if ci == co:
                return caddtable_t(h, nn.Sequential())
            else:
                return caddtable_t(h,
                            nn.Conv2d(ci, co, kernel_size=1, stride=s, padding=0, bias=False))

        def netblock(nl, ci, co, blk, s, p=0.):
            ls = [blk((i==0 and ci or co), co, (i==0 and s or 1), p) for i in range(nl)]
            return nn.Sequential(*ls)

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False),
                netblock(n, nc[0], nc[1], block, 1, d),
                netblock(n, nc[1], nc[2], block, 2, d),
                netblock(n, nc[2], nc[3], block, 2, d),
                bn2(nc[3],track_running_stats=track_running_stats),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8),
                View(nc[3]),
                nn.Linear(nc[3], num_classes))

        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, math.sqrt(2./m.in_features))
                m.bias.data.zero_()

        self.N = num_parameters(self.m)
        # s = '[%s] Num parameters: %d'%(self.name, self.N)
        # # print(s)
        # logging.info(s)

    def forward(self, x):
        return self.m(x)

opt = dict()

class WRN101(wideresnet):
    name ='WRN101'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 10,1
        super(WRN101, self).__init__(opt, num_classes=num_classes)

class WRN521(wideresnet):
    name ='WRN521'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 52,1
        super(WRN521, self).__init__(opt, num_classes=num_classes)

class WRN164(wideresnet):
    name ='WRN164'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 16,4
        super(WRN164, self).__init__(opt, num_classes=num_classes)

class WRN168(wideresnet):
    name ='WRN168'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 16,8
        super(WRN168, self).__init__(opt, num_classes=num_classes)

class WRN2810(wideresnet):
    name ='WRN2810'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 28, 10
        super(WRN2810, self).__init__(opt, num_classes=num_classes)

class WRN2812(wideresnet):
    name ='WRN2812'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 28, 12
        super(WRN2812, self).__init__(opt, num_classes=num_classes)

class WRN4010(wideresnet):
    name ='WRN4010'
    def __init__(self, num_classes=10):
        opt['depth'], opt['widen'] = 40, 10
        super(WRN4010, self).__init__(opt, num_classes=num_classes)


def wrn2810(num_classes=10):
    return WRN2810(num_classes)
    

def resnet10(num_classes=10):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)

# test()