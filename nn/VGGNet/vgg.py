import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary

#VGG-16
class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG,self).__init__()

        self.conv1 = self.conv_stack(2, in_channels, 64)
        self.conv2 = self.conv_stack(2, 64, 128)
        self.conv3 = self.conv_stack(3, 128, 256)
        self.conv4 = self.conv_stack(3, 256, 512)
        self.conv5 = self.conv_stack(3, 512, 512)
        

        self.fcn = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5) ,
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5) ,
            nn.Linear(4096,num_classes)
        )
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fcn(x), dim=1)
        return x
    

    def conv_stack(self, n, inp, out):
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
                for _ in range(n-1)
            ],
            nn.MaxPool2d(2,2)
        )


if __name__=="__main__":
    model = VGG()
    summary(model, input_size=(3,224,224))
    dummy_input = Variable(torch.randn(1,3,224,224))
    torch.onnx.export(model, dummy_input, "model.onnx")
