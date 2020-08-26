import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet,self).__init__()
        
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )

        self.part2 = nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5) ,
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5) ,
            nn.Linear(4096,num_classes)
        )
    
    def forward(self, x):
        x = self.part1(x)
        x = torch.flatten(x, 1)
        x = self.part2(x)
        x = F.softmax(x, dim=1)

        return x

if __name__=="__main__":
    model = AlexNet()
    summary(model, input_size=(3, 227,227))
    dummy_input = Variable(torch.randn(1,3, 227,227))
    torch.onnx.export(model, dummy_input, "model.onnx")
