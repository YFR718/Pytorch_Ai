import torch.nn as nn



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            # 32,32,1
            nn.Conv2d(1,6,kernel_size=5),
            # 28,28,6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 14,14,6
            nn.Conv2d(6,16,kernel_size=5),
            # 10,10,16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 5,5,16
        )
        self.liner = nn.Sequential(
            nn.Linear(5*5*16,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,10),
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.liner(x)
        return x


from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet().to(device)
summary(model, input_size=(1, 32, 32))

