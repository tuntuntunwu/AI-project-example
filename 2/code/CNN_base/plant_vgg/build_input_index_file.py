# Reference
# https://blog.csdn.net/z564359805/article/details/80805433

import os
import random

trainset = []
testset = []
path = "./yangben"
subpaths = os.listdir(path)

for i, subpath in enumerate(subpaths):
    subpath = path + "/" + subpath
    instances = os.listdir(subpath)
    # split the dataset into 0.8 & 0.2 randomly
    random.shuffle(instances)
    spoint = int(len(instances) * 0.8)
    for instance in instances[:spoint]:
        trainset.append([str(i), subpath + "/" + instance])
    for instance in instances[spoint:]:
        testset.append([str(i), subpath + "/" + instance])

# shuffling dataset is necessary
random.shuffle(trainset)
random.shuffle(testset)

# write datasets' info into files to implement data loading in PyTorch
with open("train.txt", 'w') as ftr:
    for train_instance in trainset:
        ftr.write(train_instance[1] + " " + train_instance[0] + '\n')
with open("test.txt", 'w') as fte:
    for test_instance in testset:
        fte.write(test_instance[1] + " " + test_instance[0] + '\n')
