import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from tqdm import tqdm


class BlockA(nn.Module):
    """Class for Block A which is pretty simple"""
    def __init__(self, in_channels, out_channels):
        super(BlockA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BlockB(nn.Module):
    """Class for Block B which uses Block A as base and then combines it with a conv 1x1 layer and then scales it"""
    def __init__(self, in_channels, out_channels):
        super(BlockB, self).__init__()
        self.block_a = BlockA(in_channels, out_channels)
        self.conv1 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block_a(x)
        attn = self.conv1(x)
        attn = self.sigmoid(attn)
        # here we scale the x which is the output of block A with attention we got from conv 1x1 combination
        output = x * attn

        return output


class BlockC(nn.Module):
    """Class for Block C which uses Block A as a base but then combines it with an extractive block"""
    def __init__(self, in_channels, out_channels, r=16):
        super(BlockC, self).__init__()
        self.block_a = BlockA(in_channels, out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels // r, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block_a(x)
        pooled = self.global_pool(x).view(x.size(0), -1)
        attn = self.fc1(pooled)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        attn = attn.view(x_blocka.size(0), -1, 1, 1)
        # here we scale the x which is the output of block A with attention we got from the other introduced block
        output = x * attn

        return output


class BlockD(nn.Module):
    """Class for Block D Which uses residual mechanism"""
    def __init__(self, in_channels, out_channels):
        super(BlockD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        residual = x
        # here is the residual part
        if self.in_channels != self.out_channels:
            residual = self.conv1(x)
        out += residual
        out = self.relu(out)

        return out


class DepthwiseSepD(nn.Module):
    """Class for Block D but this time it uses Depth Wise Separable mechanism to reduce parameters"""
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSepD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        residual = x
        # same residual logic
        if self.in_channels != self.out_channels:
            residual = self.conv1(x)
        out += residual
        out = self.relu(out)
        return out


class BlockE(nn.Module):
    """Class for Block E Which uses groups in Conv 3x3 and also residuals"""
    def __init__(self, in_channels, out_channels):
        super(BlockE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        b = in_channels // 4
        g = in_channels // 8
        self.conv1 = nn.Conv2d(in_channels, b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(b)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(b, b, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(b)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(b, b, kernel_size=3, padding=1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(b)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(b, out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv1_short = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        # residual logic:
        residual = self.conv1_short(residual) if self.in_channels != self.out_channels else residual
        out += residual
        out = self.relu4(out)

        return out


class Model(nn.Module):
    """Overall Model Structure which is explained by a comment to each line determining the inp and output shape"""
    def __init__(self, Block):
        super(Model, self).__init__()
        self.block1 = BlockA(3, 32)  # Block1: Input 3, Output 32
        self.block2 = BlockA(32, 64)  # Block2: Input 32, Output 64
        self.block3 = BlockA(64, 128)  # Block3: Input 64, Output 128

        self.block4 = Block(128, 128)  # Block4: Input 128, Output 128
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = Block(128, 128)  # Block5: Input 128, Output 128
        self.block6 = Block(128, 256)  # Block6: Input 128, Output 256
        self.block7 = Block(256, 256)  # Block7: Input 256, Output 256
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block8 = Block(256, 256)  # Block8: Input 256, Output 256
        self.block9 = Block(256, 512)  # Block9: Input 256, Output 512
        self.block10 = Block(512, 512)  # Block10: Input 512, Output 512
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block11 = Block(512, 512)  # Block11: Input 512, Output 512
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool2(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool3(x)

        x = self.block11(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_parameters(model):
    """Function to return model learnable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
