import gzip
import struct
import numpy as np
import time
from collections import Counter
from collections import OrderedDict
import operator

# 数据读入
# 感谢分享！！感谢分享！！感谢分享！！
def getData():
    train_img, train_label = _read("train-images-idx3-ubyte.gz",
                                "train-labels-idx1-ubyte.gz")
    test_img, test_label = _read("t10k-images-idx3-ubyte.gz",
                                "t10k-labels-idx1-ubyte.gz")
    return train_img[:6000], train_label[:6000], test_img[:500], test_label[:500
                                                                            ]                                                                          
def _read(image, label):
    mnist_dir = "data/"
    with gzip.open(mnist_dir + label, 'rb') as flbl:
        # ">"表示大端模式;"II"表示读取2个整形数;"fimg.read(8)"表示读入16个字节
        lnum, num = struct.unpack(">II", flbl.read(8))
        labels = np.frombuffer(flbl.read(), dtype=np.int8)
    with gzip.open(mnist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        # 读入num张rows*cols大小的图片,图片每个像素点用一个8位无符号整形表示
        images = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(num,
                                                                    rows, cols)
    return images, labels

# 距离计算
# 使用欧式距离
def calDistance(imatrix0, imatrix1):
    # 二值化至关重要,这样算出的欧式距离才有意义
    tmp0 = np.array(imatrix0) / 128
    tmp1 = np.array(imatrix1) / 128
    vec0 = tmp0.astype(int)
    vec1 = tmp1.astype(int)
    #dist = np.power(np.sum(np.power(vec0 - vec1, 50000)), 50000) # 切比雪夫距离
    #dist = np.sqrt(np.sum(np.square(vec0 - vec1)))  # 欧式距离
    dist = np.sum(np.abs(vec0 - vec1))  # 曼哈顿距离
    return dist

# KNN算法
def knn(k, train_img, train_label, test_img):
    if k < 0 or k > len(train_img) or type(k) != int:
        print("You input wrong k value")
        exit()
    
    predict_results = []
    for teimg in test_img:
        
        # dist_list存放 each test_img 与 all train_img 的距离
        dist_list = []
        for trimg in train_img:
            dist_list.append(round(calDistance(teimg, trimg), 2))
        
        # sorted_dict存放 已排序的有序键值对,其中key为距离,value为类别标记
        tmp_dict = {k:v for k,v in zip(dist_list, train_label)}
        sorted_dict = OrderedDict(sorted(tmp_dict.items(),
                                        key=operator.itemgetter(0)))
        # class_list存放 最近的k个距离对应的类别标记
        class_list = list(sorted_dict.values())[:k]
        print(class_list)
        
        predict_results.append(decision(class_list))
    
    return predict_results

# 分类决策规则
# 多数表决
def decision(class_list):
    # 使用python标准库collections中的Counter可以快速统计list中元素出现的次数
    # python标准库中的内容都是底层c写的,执行速度比使用python代码快很多
    tmp_class_list = [int(x) for x in class_list]
    # Counter返回个字典,key是类别标记,value是相应的个数
    cou = Counter(tmp_class_list)
    # 如果有 个数大于等于2的 类别标记则选它,如果都只有1个,则选择最近的类别标记
    if max(cou.values()) > 1:
        # 返回出现频率最高的数,形式为[(key, value)]
        cl = int(cou.most_common(1)[0][0])
    else:
        cl = tmp_class_list[0]
    
    return cl


if __name__ == '__main__':
    
    # 数据读入
    print("\n... 1.read images ...\n")
    train_img, train_label, test_img, test_label = getData()
    print("We read all images successfully!")

    # KNN算法
    print("\n... 2.KNN algorith ...\n")
    start_t = time.time()

    predict_results = knn(3, train_img, train_label, test_img)

    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We have predicted all images' label!")
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")
    
    # 计算正确率
    error_number = 0
    tmp_test_label = [int(x) for x in test_label]
    for i in range(len(predict_results)):
        if predict_results[i] != tmp_test_label[i]:
            error_number += 1
    accuracy = (1 - error_number / len(predict_results)) * 100
    print("Accuracy: " + str(accuracy) + "%")

    # 将结果写入"predict.txt"文件
    with open("./predict.txt", 'w', encoding='UTF-8') as f:
        for l in predict_results:
            f.write(str(l) + '\n')
    print("\nWe have written the predicted result in 'predict.txt'!")

    print("... end, happy to see you ...\n")
