import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary

#convolutional blocks
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#Inception module
class Inception(nn.Module):
    def __init__(self, in_channels, out1, redu3, out3, redu5, out5, pool_out):
        super(Inception,self).__init__()

        self.branch1 = conv_block(in_channels, out1, kernel_size=1)
        self.branch2 = nn.Sequential(
                        conv_block(in_channels, redu3, kernel_size=1),
                        conv_block(redu3, out3, kernel_size=3, padding=1 )
                      )
        self.branch3 = nn.Sequential(
                        conv_block(in_channels, redu5, kernel_size=1),
                        conv_block(redu5, out5, kernel_size=5, padding=2 )
                      )
        self.branch4 = nn.Sequential(
                        nn.MaxPool2d(3,1),
                        conv_block(in_channels, pool_out, kernel_size=1, padding=1)
                      )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        x = torch.cat([o1,o2,o3,o4], 1)
        return x


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogleNet,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU()
            )
        self.pool = nn.MaxPool2d(3,2)
        self.conv2 =  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        
        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,114,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)

        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.inception3a(x)
        x = self.pool(self.inception3b(x))
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.pool(self.inception4e(x))
        x = self.inception5a(x)
        x = self.avgpool(self.inception5b(x))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = F.softmax(self.classifier(x), dim=1)
        return x

if __name__=="__main__":
    model = GoogleNet()
    summary(model, input_size=(3,224,224))
    dummy_input = Variable(torch.randn(1,3,224,224))
    torch.onnx.export(model, dummy_input, "model.onnx")