import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class HybridFERModel(nn.Module):
    def __init__(self, num_classes=7, num_blocks=3):
        super(HybridFERModel, self).__init__()
        
        self.num_blocks = max(1, min(num_blocks, 3)) 
        
        # Giriş Bloğu
        self.entry_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Dinamik Bloklar için Listeler
        self.conv1_list = nn.ModuleList()
        self.conv2_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.attention_list = nn.ModuleList()
        self.res_list = nn.ModuleList()
        
        # Kanal geçişleri (Örn: 64 -> 128 -> 256 -> 512)
        channel_steps = [64, 128, 256, 512]
        
        for i in range(self.num_blocks):
            in_c = channel_steps[i]
            out_c = channel_steps[i+1]
            
            self.conv1_list.append(SeparableConv2d(in_c, out_c))
            self.conv2_list.append(SeparableConv2d(out_c, out_c))
            self.pool_list.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.attention_list.append(ChannelAttention(out_c))
            self.res_list.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_c)
            ))
            
        # Son çıkış kanal boyutu dinamik olarak belirlenir (Örn: 3 blok için 512, 1 blok için 128)
        final_channels = channel_steps[self.num_blocks]
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(final_channels, num_classes)
        )

    def forward(self, x):
        x = self.entry_conv(x)
        
        # Dinamik olarak bloklar üzerinden döngüyle geçiş
        for i in range(self.num_blocks):
            residual = self.res_list[i](x)
            out = self.conv1_list[i](x)
            out = self.conv2_list[i](out)
            out = self.pool_list[i](out)
            out = self.attention_list[i](out)
            x = out + residual
            
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x