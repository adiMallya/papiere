import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#Inception module A
class InceptionA(nn.Module):
    def __init__(self, in_channels, red5,out5, red3, out3, pool_out, out1):
        super(InceptionA,self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, red5, kernel_size=1, stride=1, padding=0),
            conv_block(red5, out5, kernel_size=3, stride=1, padding=1),
            conv_block(out5, out5, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red3, kernel_size=1, stride=1, padding=0),
            conv_block(red3, out3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3,1, padding=1),
             conv_block(in_channels, pool_out, kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        x = torch.cat([o1,o2,o3,o4], 1)
        return x

#Inception module B
class InceptionB(nn.Module):
    def __init__(self, in_channels, red7_2,out7_2, red7, out7, pool_out, out1):
        super(InceptionB,self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, red7_2, kernel_size=1, stride=1, padding=0),
            conv_block(red7_2, out7_2, kernel_size=(1,7), stride=1, padding=(0,3)),
            conv_block(out7_2, out7_2, kernel_size=(7,1), stride=1, padding=(3,0)),
            conv_block(out7_2, out7_2, kernel_size=(1,7), stride=1, padding=(0,3)),
            conv_block(out7_2, out7_2, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red7, kernel_size=1, stride=1, padding=0),
            conv_block(red7, out7, kernel_size=(1,7), stride=1, padding=(0,3)),
            conv_block(out7, out7, kernel_size=(7,1), stride=1, padding=(3,0))
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3,1,padding=1),
            conv_block(in_channels, pool_out, kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        x = torch.cat([o1,o2,o3,o4], 1)
        return x

#Inception module C
class InceptionC(nn.Module):
    def __init__(self, in_channels, red5, out5, out5_1, out5_2, red3, out3_1, out3_2, pool_out, out1):
        super(InceptionC,self).__init__()

        self.branch1 = conv_block(in_channels, red5, kernel_size=1, stride=1, padding=0)
        self.branch3x3_1 = conv_block(red5, out5, kernel_size=3, stride=1, padding=1)
        self.branch1x3_1 = conv_block(out5, out5_1, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch3x1_1 = conv_block(out5, out5_2, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2 = conv_block(in_channels, red3, kernel_size=1, stride=1, padding=0)
        self.branch1x3_2 = conv_block(red3, out3_1, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch3x1_2 = conv_block(red3, out3_2, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3,1, padding=1),
            conv_block(in_channels, pool_out, kernel_size=1, stride=1, padding=0)
        )

        self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        o1 = self.branch1(x)
        o1 = self.branch3x3_1(o1)
        o1_1 = self.branch1x3_1(o1)
        o1_2 = self.branch3x1_1(o1)
        o2 = self.branch2(x)
        o2_1 = self.branch1x3_2(o2)
        o2_2 = self.branch3x1_2(o2)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        x = torch.cat([o1_1,o1_2,o2_1,o2_2,o3,o4], 1)
        return x

#Grid size reduction module
class GridRedu(nn.Module):
    def __init__(self, in_channels, red):
        super(GridRedu,self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, red, kernel_size=1,  stride=1, padding=0),
            conv_block(red, red, kernel_size=3, stride=1, padding=1),
            conv_block(red, red, kernel_size=3, stride=2, padding=0)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red, kernel_size=1,  stride=1, padding=0),
            conv_block(red, red, kernel_size=3, stride=2, padding=0)
        )

        self.pool = nn.MaxPool2d(3,2)

    def forward(self,x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.pool(x)
        x = torch.cat([o1,o2,o3], 1)
        return x


class InceptionV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(InceptionV3,self).__init__()

        self.part1 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=3, stride=2, padding=0),
            conv_block(32, 32, kernel_size=3, stride=1, padding=0),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3,2, padding=1),
            conv_block(64, 80, kernel_size=3, stride=1, padding=0),
            conv_block(80, 192, kernel_size=3, stride=2, padding=0),
            conv_block(192, 288, kernel_size=3, stride=1, padding=1)
        )

        self.InceptionA_1 = InceptionA(288,16,32,96,128,32,64)
        self.InceptionA_2 = InceptionA(256,32,96,128,192,64,128)
        self.InceptionA_3 = InceptionA(480,32,64,128,160,64,96)

        self.Grid1 = GridRedu(384,192)

        self.InceptionB_1 = InceptionB(768,16,48,96,208,64,192)
        self.InceptionB_2 = InceptionB(512,24,64,112,224,64,160)
        self.InceptionB_3 = InceptionB(512,24,64,128,256,64,128)
        self.InceptionB_4 = InceptionB(512,32,64,144,288,64,112)
        self.InceptionB_5 = InceptionB(528,32,80,160,272,80,208)
        
        self.Grid2 = GridRedu(640,320)

        self.InceptionC_1 =  InceptionC(1280,32,64,64,64,160,160,160,128,256)
        self.InceptionC_2 =  InceptionC(832,48,64,128,128,192,384,384,256,768)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(2048,num_classes)

    def forward(self, x):
        x = self.part1(x)
        x = self.InceptionA_1(x)
        x = self.InceptionA_2(x)
        x = self.InceptionA_3(x)
        x = self.Grid1(x)
        x = self.InceptionB_1(x)
        x = self.InceptionB_2(x)
        x = self.InceptionB_3(x)
        x = self.InceptionB_4(x)
        x = self.InceptionB_5(x)
        x = self.Grid2(x)
        x = self.InceptionC_1(x)
        x = self.InceptionC_2(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc(x), dim=1)
        return x


if __name__=="__main__":
    model = InceptionV3()
    summary(model, input_size=(3,229,229))
    dummy_input = Variable(torch.randn(1,3,229,229))
    torch.onnx.export(model, dummy_input, "model.onnx")