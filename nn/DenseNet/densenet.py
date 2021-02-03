import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block,self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        
    def forward(self,x):
        return self.conv(F.relu(self.bn(x)))

    
class BottleNeck(nn.Module):
    
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck,self).__init__()

        inner_channel = 4*growth_rate

        self.bottle_neck = nn.Sequential(
            conv_block(in_channels, inner_channel, kernel_size=1),
            conv_block(inner_channel, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self,x):
        return torch.cat([x, self.bottle_neck(x)],1)


class Transition(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Transition,self).__init__()

        self.downsample = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2,2)
        )

    def forward(self,x):
        return self.downsample(x)



class DenseNet(nn.Module):
    
    def __init__(self, block, num_blocks, growth_rate=32, reduction=0.5, num_class=1000):
        super(DenseNet,self).__init__()

        self.growth_rate = growth_rate
        inner_channels = 2*growth_rate

        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.layers = nn.Sequential()

        for i in range(len(num_blocks)-1):
            self.layers.add_module(f"dense_block{i}", self._make_layers(block, inner_channels, num_blocks[i]))
            inner_channels += growth_rate*num_blocks[i]
            out_channels = int(reduction*inner_channels)
            self.layers.add_module(f"transition_block{i}", Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.layers.add_module(f'dense_block{len(num_blocks)-1}', self._make_layers(block, inner_channels, num_blocks[len(num_blocks)-1]))
        inner_channels += growth_rate*num_blocks[len(num_blocks)-1]
        self.layers.add_module(f'bn', nn.BatchNorm2d(inner_channels))
        self.layers.add_module(f'relu', nn.ReLU())


        self.pool2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(inner_channels, num_class)


    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.layers(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layers(self, block, in_channels, num_blocks):
        dense_block = nn.Sequential()
        for i in range(num_blocks):
            dense_block.add_module(f'bottle_neck{i}', block(in_channels, self.growth_rate))
            in_channels += self.growth_rate

        return dense_block

if __name__=="__main__":
    densenet_121 = DenseNet(BottleNeck, [6,12,24,16])
    summary(densenet_121, input_size=(3,224,224))
    dummy_input = Variable(torch.randn(1,3,224,224))
    torch.onnx.export(densenet_121, dummy_input, "densenet_121.onnx")