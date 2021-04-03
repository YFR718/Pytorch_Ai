import torch
from torch import nn


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Liner = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128 * 3 * 3),
            nn.ReLU(inplace=True),
            nn.Linear(128 * 3 * 3, 128 * 3 * 3),
            nn.ReLU(inplace=True),
            nn.Linear(128 * 3 * 3, 10)
        )

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.Liner(x)
        print(x.shape)
        return x


from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Alexnet().to(device)
summary(model, input_size=(1, 32, 32))
