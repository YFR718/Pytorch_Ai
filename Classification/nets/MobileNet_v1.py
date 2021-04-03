import torch
from torch import nn

def conv_bn(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )


def conv_dw(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class MobileNet_v1_25(nn.Module):
    def __init__(self):
        super(MobileNet_v1_25, self).__init__()
        self.stage1 = nn.Sequential(
            # 224,224,3
            conv_bn(3, 32, 2),
            # 112,112,32
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            # 56,56,128
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            # 28,28,256
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            # 14,14,512
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            # 7,7,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        #1,1,1024
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNet_v1_25().to(device)
summary(model, input_size=(3, 640, 640))

