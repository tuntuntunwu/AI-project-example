from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import re
import pandas as pd
import numpy as np
from numpy import transpose
import time

# 数据集处理

# 分词,去除停用词,返回存储了 文档ID 与 对应内容 的字典
def preprocess():
    # sentence_dict字典存储 文档ID 与 其对应的文本内容 
    sentence_dict = {}
    if os.path.exists("./sentence_dict.txt"):
        with open("./sentence_dict.txt", encoding='UTF-8') as f:
            for line in f.readlines():
                tmp = line.split()
                sentence_dict[tmp[0]] = tmp[1:]
    else:
        swfile = open("./stopwords.txt", encoding='UTF-8')
        stopwords = swfile.read().splitlines()
    
        line_gen = readLine("./documents.txt")
        for line in line_gen:
            tmp_re = re.search(r'<article_id>(.*)</article_id>.*<title>(.*)</title>.*<body>(.*)</body>', line, flags=0)
            if tmp_re is not None:
                docID = tmp_re.group(1)
                title = tmp_re.group(2)
                body = tmp_re.group(3)
                if not body.startswith(' '+title):
                    body = title + body
                tmp_body = body.split()
            
                for i in range(len(tmp_body)-1, -1, -1):
                    if ((len(tmp_body[i])<3) or (tmp_body[i] in stopwords)):
                        tmp_body.pop(i)
                sentence_dict[docID] = tmp_body
        
        with open("./sentence_dict.txt", 'w', encoding='UTF-8') as f:
            for k,v in sentence_dict.items():
                f.write(k + " ")
                for vi in v:
                    f.write(vi + " ")
                f.write('\n')

    return sentence_dict
#使用生成器来读取一个很大的文件
def readLine(file):
    with open(file, 'r', encoding='UTF-8') as f:
        n = 0
        # 如果你使用32位python解释器,将无法处理全部的file内容,请限制n<6100000读取最多内容
        while True:
            n += 1
            print(n)
            l = f.readline()
            if l:
                yield l
            else:
                return

# 生成向量相关函数
# 生成能将所有文档内容转化为文档向量的模型(使用gensim.models模块中的Doc2Vec生成)
# 应当针对不同的query生成不同的模型
def getDoc2Vec(sentence_list):
    documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(sentence_list)]
    model = Doc2Vec(documents, vector_size=100, min_count=1, workers=8)
    return model
# 划分数据集,返回所有将来可能要用到的 特征数据all_x 和 标签all_y 来模拟主动学习
def splitDataSet(QID, all_data, sentence_dict):
    start = (QID - 201) * 10000
    end = start + 10000
    docid_list = list(all_data['DocID'])
    # 生成能将文档内容变为文档向量的模型Doc2Vec
    query_sentence_list = []
    for _id in docid_list[start:end]:
        if sentence_dict.get(_id) is not None:
            query_sentence_list.append(sentence_dict[_id])
    model = getDoc2Vec(query_sentence_list)

    x_train = []
    for i in range(start, start+400):
        tmp_id = docid_list[i]
        # 我们忽略那些不能匹配到内容的文档.也因此,我们得多取一些数据集以取到能用的300个训练集
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            tmp_v = [1.0]
            tmp_v.extend(list(vector))
            x_train.append(tmp_v)
        else:
            continue
    n_pos = len(x_train)
    for i in range(end-400, end):
        tmp_id = docid_list[i]
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            tmp_v = [1.0]
            tmp_v.extend(list(vector))
            x_train.append(tmp_v)
        else:
            continue    
    y_train = [1] * n_pos
    y_train.extend([0] * (len(x_train) - n_pos))
    all_x = x_train
    all_y = y_train
    mid = n_pos

    return all_x, all_y, mid


# 学习器学习 - 使用Logistic Regression学习
# Logistic Regression算法
def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))
def gradAscent(dataList, labelList):
    # 梯度上升得到theta(vector)
    dataMatrix = np.mat(dataList)
    labelMatrix = np.mat(labelList).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.1  # 学习率
    theta = np.random.rand(n, 1)

    cycle_number = 0  # loop number
    gra_abs = 1
    while gra_abs > 0.01 and cycle_number < 10000:
        sv = sigmoid(dataMatrix * theta)
        error = sv - labelMatrix
        gradient = dataMatrix.transpose() * error
        theta = theta - alpha * gradient
        gra_abs = max(gradient)
        cycle_number += 1
    return theta


# 查询策略(Active Learning算法)相关函数
# 计算正确率
def calError(predict_results, y_test):
    error_number = 0
    for i in range(len(predict_results)):
        if predict_results[i] != y_test[i]:
            error_number += 1
    accuracy = (1 - error_number / len(predict_results)) * 100
    print("QID:" + str(xx) + " Accuracy:" + str(accuracy) + '%')
    return accuracy


if __name__ == '__main__':
    # 数据集处理
    print("\n... 1.preprocess data ...\n")
    
    all_data = pd.read_table("./Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",
                        delimiter=' ', header=None, names=["QID", "x0", "DocID",
                        "score", "relevancy", "x1"])
    sentence_dict = preprocess()
    
    print("We preprocess all data successfully!")

    # Active Learning算法
    # 查询策略实现部分 - 使用 不确定性采样 查询策略
    print("\n... 2.Active Learning algorith ...\n")
    start_t = time.time()

    # qsize存储10个queries达到0.75以上正确率时,当前已标注数据集的大小
    qsize = []
    for xx in range(201, 211):
        # 初始化
        all_x, all_y, mid = splitDataSet(xx, all_data, sentence_dict)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        # 以 10个正类,10个负类的样例 进行初始的训练
        x_train = all_x[mid-10:mid+10]
        y_train = all_y[mid-10:mid+10]
        # i_list维护着未标记数据的索引值
        i_list = []
        i_list.extend(range(0, mid-10))
        i_list.extend(range(mid+10, len(all_x)))
        i_list = np.array(i_list)
        
        while True:
            x_test = all_x[i_list]
            print("unlabeled_number:" + str(len(x_test)))
            y_test = all_y[i_list]
        
            #预测结果
            T = gradAscent(x_train, y_train)
            all_value_results = sigmoid(np.mat(all_x) * T)
            predict_results = []
            for res in all_value_results[i_list]:
                if res > 0.5:
                    predict_results.append(1)
                else:
                    predict_results.append(0)
            # 计算准确率
            accuracy = calError(predict_results, all_y[i_list])
        
            if accuracy > 75:
                # 准确率到达0.75时跳出
                print("\naccuracy break\n")
                qsize.append(len(i_list))
                break
            else:
                nn = 0
                # 采用不确定性采样进行样本选择、标注
                # 将0.4< <0.6的进行标注
                for i in range(len(all_value_results)):
                    if all_value_results[i] < 0.6 and all_value_results[i] > 0.4\
                        and i in i_list:
                        nn += 1
                        i_list = list(i_list)
                        # 更新、标注样例
                        i_list.remove(i)
                        x_train = list(x_train)
                        x_train.append(all_x[i])
                        y_train = list(y_train)
                        y_train.append(all_y[i])
                # 如果不存在0.4< <0.6,对0.3< <0.7的进行标注
                if nn == 0:
                    for i in range(len(all_value_results)):
                        if all_value_results[i] < 0.7 and\
                            all_value_results[i] > 0.3 and i in i_list:
                            nn += 1
                            i_list = list(i_list)
                            # 更新、标注样例
                            i_list.remove(i)
                            x_train = list(x_train)
                            x_train.append(all_x[i])
                            y_train = list(y_train)
                            y_train.append(all_y[i])
                # 如果不存在0.3< <0.7,对0.2< <0.8的进行标注
                if nn == 0:
                    for i in range(len(all_value_results)):
                        if all_value_results[i] < 0.8 and\
                            all_value_results[i] > 0.2 and i in i_list:
                            nn += 1
                            i_list = list(i_list)
                            # 更新、标注样例
                            i_list.remove(i)
                            x_train = list(x_train)
                            x_train.append(all_x[i])
                            y_train = list(y_train)
                            y_train.append(all_y[i])
                # 如果不存在0.2< <0.8,对0.1< <0.9的进行标注
                if nn == 0:
                    for i in range(len(all_value_results)):
                        if all_value_results[i] < 0.9 and\
                            all_value_results[i] > 0.1 and i in i_list:
                            nn += 1
                            i_list = list(i_list)
                            # 更新、标注样例
                            i_list.remove(i)
                            x_train = list(x_train)
                            x_train.append(all_x[i])
                            y_train = list(y_train)
                            y_train.append(all_y[i])
                # 如果不存在0.1< <0.9,对剩下全部的数据进行标注
                if nn == 0:
                    print("\nall labeled break\n")
                    qsize.append(len(all_x))
                    break

    # qsize存储10个queries达到0.75以上正确率时,当前已标注数据集的大小
    print("10 queries' labeled_number:")
    print(qsize)
    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s to\
        simulate the active learning algorith")
    with open("./predict.txt", 'w', encoding='UTF-8') as f:
            f.write(str(qsize))
    print("\nWe have written the predicted result in 'predict.txt'!")
    
    print("\n... end, happy to see you ...\n")
