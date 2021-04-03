import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            # 32,32,1
            nn.Conv2d(1, 6, kernel_size=5),
            # 28,28,6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 14,14,6
            nn.Conv2d(6, 16, kernel_size=5),
            # 10,10,16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 5,5,16
        )
        self.liner = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.liner(x)
        return x


batch_size = 512
learning_rate = 0.01
EPOCH = 100
# 若检测到GPU环境则使用GPU，否则使用CPU。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 导入数据
train_data = datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.CenterCrop((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.2860,), (0.3081,))
                            ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.CenterCrop((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.2860,), (0.3081,))
                           ]))

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# 构建网络
net = LeNet().to(device)
print(net)
loss_fuc = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练
train_loss = []
train_acc = []
test_acc = []
for epoch in range(EPOCH):
    sum_loss = 0
    # 数据读取
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True):
        x = x.to(device)
        y = y.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播，计算损失梯度，更新参数
        output = net(x)
        # y_onehot = one_hot(y)
        # loss = F.mse_loss(output, y_onehot)
        loss = loss_fuc(output, y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # print(epoch, "loss:", loss.item())

    correct = 0
    total = 0
    for ss, (tx, ty) in enumerate(train_loader):
        tx, ty = tx.to(device), ty.to(device)
        outputs_test = net(tx)
        _, predicted = torch.max(outputs_test.data, 1)  # 输出得分最高的类
        total += ty.size(0)  # 统计所有batch图片的总个数
        correct += (predicted == ty).sum()  # 统计正确分类的个数
    train_acc.append(float(100.0 * correct / total))
    print('第%d个epoch训练集准确率为：%f%%' % (epoch + 1, (100.0 * correct / total)))

    correct = 0
    total = 0
    for ss, (tx, ty) in enumerate(test_loader):
        tx, ty = tx.to(device), ty.to(device)
        outputs_test = net(tx)
        _, predicted = torch.max(outputs_test.data, 1)  # 输出得分最高的类
        total += ty.size(0)  # 统计50个batch 图片的总个数
        correct += (predicted == ty).sum()  # 统计50个batch 正确分类的个数
    test_acc.append(float(100.0 * correct / total))
    print('第%d个epoch测试集准确率为：%f%%' % (epoch + 1, (100.0 * correct / total)))

print("最大测试集准确度", max(test_acc))
print("最大训练集准确度", max(train_acc))

plt.plot(range(1, EPOCH + 1), train_acc, label="train_acc")
plt.plot(range(1, EPOCH + 1), test_acc, label="test_acc")
plt.legend()  ##显示上面的label
plt.show()
print("最大测试集准确度", max(test_acc))
