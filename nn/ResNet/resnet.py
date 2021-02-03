import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary

class BasicBlock(nn.Module):

    expansion=1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()


        #residual blokc
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )

        #skip connection
        self.skip = nn.Sequential()

        #projection connection
        if stride != 1 or in_channels != out_channels*BasicBlock.expansion:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )

    def forward(self,x):
        return F.relu(self.residual_block(x)+self.skip(x))


#Bottle-necks for resnets 50/101/152
class BottleNeck(nn.Module):

    expansion=4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck,self).__init__()


        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion),
        )

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels*BottleNeck.expansion:
                    self.skip = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride),
                        nn.BatchNorm2d(out_channels*BottleNeck.expansion)
                    )
    
    def forward(self,x):
        return F.relu(self.residual_block(x)+self.skip(x))



class ResNet(nn.Module):
    def __init__(self, block_type, num_layers, num_classes=1000):
        super(ResNet,self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2_x = self.make_layers(block_type, 64, num_layers[0], 1)
        self.conv3_x = self.make_layers(block_type, 128, num_layers[1], 2)
        self.conv4_x = self.make_layers(block_type, 256, num_layers[2], 2)
        self.conv5_x = self.make_layers(block_type, 512, num_layers[3], 2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512*block_type.expansion, num_classes)


    def make_layers(self, block_type, out_channels, num_layers, stride):
        layers = []

        strides = [stride] + [1] + [num_layers-1]

        for stride in strides:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block_type.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__=="__main__":
    resnet50 = ResNet(BottleNeck, [3, 4, 6, 3])
    summary(resnet50, input_size=(3,224,224))
    dummy_input = Variable(torch.randn(1,3,224,224))
    torch.onnx.export(resnet50, dummy_input, "resnet50.onnx")