import os, json, random

root0 = "../input/data/6"
dirs = os.listdir(root0)
print(dirs)
print(len(dirs))
with open("./classes.json", 'w', encoding='utf-8') as f:
    json.dump(dirs, f)

trainset = list()
testset = list()
for i, di in enumerate(dirs):
    tmp_dir = os.path.join(root0, di)
    files = os.listdir(tmp_dir)
    
    # split all data into 0.8 & 0.2 randomly
    random.shuffle(files)
    spoint = int(len(files) * 0.8)
    for file_name in files[:spoint]:
        trainset.append(str(i) + ' ' + os.path.join(tmp_dir, file_name))
    for file_name in files[spoint:]:
        testset.append(str(i) + ' ' + os.path.join(tmp_dir, file_name))

# shuffling dataset is necessary
random.shuffle(trainset)
random.shuffle(testset)

# write datasets' info to files to implement data loading in PyTorch
trainset_file = "./train11.txt"
testset_file = "./test11.txt"

with open(trainset_file, 'w', encoding='utf-8') as ftr:
    for train_instance in trainset[:-1]:
        ftr.write(train_instance + '\n')
    ftr.write(trainset[-1])
with open(testset_file, 'w', encoding='utf-8') as fte:
    for test_instance in testset[:-1]:
        fte.write(test_instance + '\n')
    fte.write(testset[-1])


import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time, os, json


# GPU1: set GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
        # TODO
        #self.classifier = nn.Linear(2048, class_number)
        self.classifier = nn.Linear(8192, class_number)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        # TODO
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # BatchNorm2d() is helpful
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

# Dataset
root = "./"
with open("./classes.json", 'r', encoding='utf-8') as f:
    classes = json.load(f)
    classes = tuple(classes)
class_number = len(classes)
print(classes)
print(class_number)
model_path = "./jinwen_vgg11.pt"

class OwnData(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        fh = open(os.path.join(root, datatxt), 'r', encoding='utf-8')  # read index files related to dataset to load them
        imgs = []
        for line in fh.readlines():
            words = line.strip().split()
            img_path = ""
            for fragment in words[1:]:
                img_path += fragment + " "
            img_path = img_path.strip()
            label = int(words[0])
            imgs.append((img_path, label))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img_path, label = self.imgs[index]  # image path and image label
        img = Image.open(img_path).convert('RGB')  # convert to binary images

        # preprocess image data
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # 1.Load and normalizing the training and test datasets using torchvision.transforms
    # TODO
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = OwnData(root=root, datatxt='train11.txt', transform=transform)
    # this dataset size is irregular, so we just have to feed only 1 image in each batch
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=4)

    testset = OwnData(root=root, datatxt='test11.txt', transform=transform)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4)
            
    # 2.Define a Convolutional Neural Network
    if os.path.exists(model_path):
        # load trained net which has existed
        net = VGG('VGG11')
        net.load_state_dict(torch.load(model_path))
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
        # TODO
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # 4.Train the network on the training data
        start_t = time.time()

        itr_number = 5
        for epoch in range(itr_number):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):  # return a batch of data
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
            
            with torch.no_grad(): 
                # classes' accuracy
                class_correct = list(0. for i in range(class_number))
                class_total = list(0. for i in range(class_number))
                for (inputs, labels) in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    c = (predicted == labels).squeeze().cpu().numpy()
                    # calculate every batch
                    for i in range(1):
                        label = labels[i]
                        #class_correct[label] += c[i]
                        class_correct[label] += c
                        class_total[label] += 1

                for i in range(class_number):
                    if class_total[i]:
                        print('Accuracy of %5s : %.4f %%' % (
                            classes[i], 100 * class_correct[i] / class_total[i]))

                # total accuracy
                correct = 0
                total = 0
                for i in range(class_number):
                    correct += class_correct[i]
                    total += class_total[i]
                print('Accuracy of the network on test images: %.4f %%' % (
                    100 * correct / total))
                
                torch.save(net.state_dict(), os.path.join(root, "itr" + str(epoch) + "_jinwen_vgg11.pt"))

        print('Finished Training')

        end_t = time.time()
        m, s = divmod(end_t-start_t, 60)
        print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s to train VGG!\n")
    