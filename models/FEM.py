import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# FEM模块（修复版）
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=4):
        super(FEM, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.out_channels = out_planes
        inter_planes = max(in_planes // map_reduce, 1)  # 确保 inter_planes ≥1

        # Branch0: 基础特征提取
        min_channels_branch0 = max(2 * inter_planes, 8)
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, min_channels_branch0, 1, stride),
            BasicConv(min_channels_branch0, 2 * inter_planes, 3, 1, padding=1, relu=False)
        )

        # Branch1: 垂直方向特征（修复中间通道数计算）
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, 1, 1),
            BasicConv(inter_planes, max((inter_planes // 2) * 3, 1), 1, 1, padding=(0, 1)),  # 关键修复
            BasicConv(max((inter_planes // 2) * 3, 1), 2 * inter_planes, 3, 1, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, 3, 1, padding=5, dilation=5, relu=False)
        )

        # Branch2: 水平方向特征（修复中间通道数计算）
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, 1, 1),
            BasicConv(inter_planes, max((inter_planes // 2) * 3, 1), 3, 1, padding=(1, 0)),  # 关键修复
            BasicConv(max((inter_planes // 2) * 3, 1), 2 * inter_planes, 1, 1, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, 3, 1, padding=5, dilation=5, relu=False)
        )

        # 特征融合层
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, 1, 1, relu=False)
        self.shortcut = nn.Sequential(
            BasicConv(in_planes, max(16, in_planes // 2), 1, stride, relu=False),
            BasicConv(max(16, in_planes // 2), out_planes, 1, 1, relu=False)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat([x0, x1, x2], dim=1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        return self.relu(out)

# ======================
# 测试代码（直接运行验证）
# ======================
if __name__ == "__main__":
    # 测试输入 (Batch=1, Channels=3, Height=64, Width=64)
    x = torch.randn(1, 3, 64, 64)

    # 初始化模型 (in_planes=3, out_planes=64)
    model = FEM(in_planes=3, out_planes=64, map_reduce=4)

    # 前向传播
    output = model(x)

    # 验证输出形状
    print("Output shape:", output.shape)  # 预期输出: torch.Size([1, 64, 64, 64])