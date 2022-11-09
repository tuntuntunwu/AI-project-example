import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv1 表输入的特征数为1，输出为特征数为6，卷积核大小3*3
        # conv2 表述 入的特征数为6，输出为特征数为12，卷积核大小3*3
        # 其余依此推类
        # pool 为最大池化 大小为2*2 实际操作会使得特征映射的长宽尺寸为原先的1/2，数量不变
        # fc1 为全连接层 计算中48为特征映射数，4*4表示一个特征映射的大小，120是这层的神经元数量
        # fc2和fc3 为全连接层 输入和输出同fc1
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6,12,3)
        self.conv3 = nn.Conv2d(12,24,3)
        self.conv4 = nn.Conv2d(24,48,3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 同__init__中注释
        # 此函数表示
        # 首先输入经过一个卷积层，再通过一个relu激活函数得到输出
        # 输出再经过一个卷积层，再通过一个relu激活函数得到输出
        # 最后输出经过一个池化层后拉平，进入三个全连连接层
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 48 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = Net()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    trainset = dset.MNIST('', train=True, transform=transform, target_transform=None, download=True)
    testset = dset.MNIST('', train=False, transform=transform, target_transform=None, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(10):  # loop over the dataset multiple times
        if epoch == 7:
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        running_loss = 0.0
        for i,  (inputs, labels) in enumerate(trainloader, 0):
            # wrap them in Variable
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    with torch.no_grad(): 
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze().numpy()
            #print(c)
            for i in range(4):
                label = labels[i]
                #print(label)
                #input()
                class_correct[label] += c[i]
                class_total[label] += 1
            
        for i in range(10):
            print('Accuracy of %5s : %.4f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.4f %%' % (
            100 * correct / total))
        #input()

if __name__ == '__main__':
    main()