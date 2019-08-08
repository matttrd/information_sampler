import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 

class ModulatedAttLayer(nn.Module):
    def __init__(self, in_channels, reduction = 2, mode='embedded_gaussian'):
        super(ModulatedAttLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size = 1)
        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size = 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        # self.fc_channel = nn.Linear(7 * 7 * self.in_channels, self.in_channels)
        # self.fc_selector = nn.Linear(7 * 7 * self.in_channels, 1)

        # self.triplet_loss = TripletLoss(margin=0.2)

        self.init_weights()

    def init_weights(self):
        msra_list = [self.g, self.theta, self.phi]
        for m in msra_list:
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()

    def embedded_gaussian(self, x):
        # embedded_gaussian cal self-attention, which may not strong enough
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)
        
        x_flatten = x.view(-1, 7 * 7 * self.in_channels)
        
        # channel_att = self.fc_channel(x_flatten)
        # channel_att = channel_att.softmax(dim=1)

        spatial_att = self.fc_spatial(x_flatten)
        spatial_att = spatial_att.softmax(dim=1)

        # selector = self.fc_selector(x_flatten)
        # selector = selector.sigmoid()
        
        # class_count = torch.LongTensor(class_count)
        # class_type = torch.ones_like(class_count)
        # class_type[class_count > 100] = 0
        # class_type[class_count < 20] = 2
        # labels_type = class_type[labels]
        
        # loss_attention = self.triplet_loss(channel_selector, labels_type)
        # loss_attention = 0.0

        # channel_att = channel_att.unsqueeze(2).unsqueeze(3).expand(-1, -1, 7, 7)
        
        spatial_att = spatial_att.view(-1, 7, 7).unsqueeze(1)
        spatial_att = spatial_att.expand(-1, self.in_channels, -1, -1)

        # selector = selector.unsqueeze(2).unsqueeze(3).expand(-1, self.in_channels, 7, 7)

        # final = spatial_att * channel_att * mask + x
        final = spatial_att * mask + x

        # return final, loss_attention
        return final, [x, spatial_att, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplementedError 
        return output, feature_maps

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, use_modulatedatt=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
  
        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            print('Using self attention.')
            self.modulatedatt = ModulatedAttLayer(in_channels=512*block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_modulatedatt:
            x, feature_maps = self.modulatedatt(x)
        else:
            feature_maps = None

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_maps


def resnet10(num_classes=1000, use_att=False):
    return ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes, use_modulatedatt=use_att)

def resnet18(num_classes=1000, use_att=False):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_modulatedatt=use_att)

def resnet34(num_classes=1000, use_att=False):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, use_modulatedatt=use_att)

def resnet50(num_classes=1000, use_att=False):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_modulatedatt=use_att)

def resnet101(num_classes=1000, use_att=False):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_modulatedatt=use_att)

def resnet152(num_classes=1000, use_att=False):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_modulatedatt=use_att)