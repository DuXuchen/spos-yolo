import numpy as np
import torch
import torch.nn as nn

from models.block import Choice_Block, Choice_Block_x
from models.yolo import Detect

channel = [16,
           64, 64, 64, 64,
           160, 160, 160, 160,
           320, 320, 320, 320, 320, 320, 320, 320,
           640, 640, 640, 640]
last_channel = 1024


class SinglePath_OneShot(nn.Module):
    def __init__(self, dataset, resize, layers, nc, anchors):
        super(SinglePath_OneShot, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0], affine=False),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Block_x(inp, oup, stride=stride))
                else:
                    layer_cb.append(Choice_Block(inp, oup, kernel=j, stride=stride))
            self.choice_block.append(layer_cb)
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel, affine=False),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)

        # 检测头
        self.detect_head = Detect(nc=nc, anchors=anchors, ch=channel)
        self._initialize_weights()

    def forward(self, x, choice=None):
        """Forward with an explicit `choice` list or sample one randomly when None.

        Args:
            x (Tensor): input images
            choice (iterable[int] or Tensor, optional): per-layer choice indices. If None,
                a random choice of length `self.layers` is sampled uniformly from
                [0, len(self.kernel_list)).
        """
        # sample a random architecture if not provided
        if choice is None:
            choice = np.random.randint(len(self.kernel_list), size=self.layers)

        # normalize choice to a Python sequence of ints
        if isinstance(choice, torch.Tensor):
            choice = choice.tolist()
        elif isinstance(choice, np.ndarray):
            choice = choice.tolist()

        x = self.stem(x)
        # repeat through choice blocks
        for i, j in enumerate(choice):
            x = self.choice_block[i][int(j)](x)
        x = self.last_conv(x)

        # Detection head
        detection_output = self.detect_head(x)
        return detection_output

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SinglePath_Network(nn.Module):
    def __init__(self, dataset, resize, layers, choice, nc, anchors):
        super(SinglePath_Network, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]

        self.layers = layers
        self.kernel_list = [3, 5, 7, 'x']

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            if choice[i] == 3:
                self.choice_block.append(Choice_Block_x(inp, oup, stride=stride, supernet=False))
            else:
                self.choice_block.append(Choice_Block(inp, oup, kernel=self.kernel_list[choice[i]], stride=stride, supernet=False))
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        
        # 检测头
        self.detect_head = Detect(nc=nc, anchors=anchors, ch=channel)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        # repeat
        for i in range(self.layers):
            x = self.choice_block[i](x)
        x = self.last_conv(x)
        # 进行目标检测
        detection_output = self.detect_head(x)
        return detection_output

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
