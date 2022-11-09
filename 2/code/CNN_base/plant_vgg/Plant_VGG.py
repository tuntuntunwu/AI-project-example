# Reference

# net
# https://pytorch.org/tutorials/index.html

# VGG
# https://blog.csdn.net/qq_16234613/article/details/79818370
# https://blog.csdn.net/Codeur/article/details/78057714

# load your own dataset
# https://www.jb51.net/article/140472.htm
# https://blog.csdn.net/Teeyohuang/article/details/79587125

# preprocess image data
# https://blog.csdn.net/u014380165/article/details/79167753
# https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-transform/

# Q&A
# https://oldpan.me/archives/pytorch-conmon-problem-in-training

import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time
import os

# GPU1: set GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # features
        self.features = self._make_layers(cfgs[vgg_name])
        # classifier
        self.classifier = nn.Linear(2048, 16)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # BatchNorm2d() is helpful
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

# Plant Dataset
class PlantData(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        fh = open(root + datatxt, 'r')  # read index files related to dataset to load them
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        fn, label = self.imgs[index]  # image path and image label
        img = Image.open(self.root+fn).convert('RGB')  # transform to RGB images including 3 channels

        # preprocess image data
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.imgs)


def main():
    # 1.Load and normalizing the training and test datasets using torchvision.transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=2),
        transforms.ToTensor(),        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = PlantData(root='', datatxt='./train.txt', transform=transform)
    # dataset is splited to 0.8&0.2 whose number is irregular, so we can only feed 1 image in each batch
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=8)

    testset = PlantData(root='', datatxt='./test.txt', transform=transform)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=8)

    classes = ('bajiaojinpan', 'changchunteng', 'chuiliu', 'dongqing', 'gangzhu',
           'guangyulan', 'jizhuaqi', 'luhui', 'luwei', 'shuishan', 'tao',
           'wucaisu', 'yinxing', 'yueji', 'yulan', 'zonglv')
            
    # 2.Define a Convolutional Neural Network
    if os.path.exists("./Plant_vgg_12itr.pt"):
        # load trained net which has existed
        net = VGG('VGG11')
        net.load_state_dict(torch.load("./Plant_vgg_12itr.pt"))
        net.eval()
    else:
        net = VGG('VGG11')
        # Data Parallelism: one sentence is enough!
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        # GPU2: push net into GPU
        net.to(device)

        # 3.Define a loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

        # 4.Train the network on the training data
        start_t = time.time()

        for epoch in range(12):  # loop over the dataset multiple times
            running_loss = 0.0
            for i,  (inputs, labels) in enumerate(trainloader, 0):
                # GPU3: push data into GPU
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches' loss
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), "./Plant_vgg_12itr.pt")

        end_t = time.time()
        m, s = divmod(end_t-start_t, 60)
        print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s to train VGG!\n")
    
    # 5.Test the network on the test data
    with torch.no_grad(): 
        # classes' accuracy
        class_correct = list(0. for i in range(16))
        class_total = list(0. for i in range(16))
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze().numpy()
            # calculate every batch
            for i in range(1):
                label = labels[i]
                #class_correct[label] += c[i]
                class_correct[label] += c
                class_total[label] += 1
            
        for i in range(16):
            print('Accuracy of %5s : %.4f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        
        # total accuracy
        correct = 0
        total = 0
        for i in range(16):
            correct += class_correct[i]
            total += class_total[i]

        print('Accuracy of the network on the 10000 test images: %.4f %%' % (
            100 * correct / total))

if __name__ == '__main__':
    main()