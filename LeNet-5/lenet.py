import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
import tensorflow as tf 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.part1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(2,2)
        )

        self.part2 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x = self.part1(x)
        x = x.view(x.shape[0], -1)
        x = self.part2(x)
        x = F.softmax(x, dim=1)

        return x


if __name__=="__main__":
    model = LeNet()
    summary(model, input_size=(1, 32,32))
    torch.save(model, "lenet.pt")
    