from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import re
import pandas as pd
import numpy as np
from numpy import transpose
import time

# 预处理数据相关函数

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
    
        line_gen = readLine("./documents.txt", encoding='UTF-8')
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
# 划分数据集,返回x_train, y_train, x_test, y_test
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
    for i in range(start+100, start+400):
        tmp_id = docid_list[i]
        # 我们忽略那些不能匹配到内容的文档ID
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            tmp_v = [1.0]
            tmp_v.extend(list(vector))
            x_train.append(tmp_v)
        else:
            continue
    n_pos = len(x_train)
    for i in range(end-400, end-100):
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

    x_test = []
    for i in range(start, start+100):
        tmp_id = docid_list[i]
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            tmp_v = [1.0]
            tmp_v.extend(list(vector))
            x_test.append(tmp_v)
        else:
            continue
    n_pos = len(x_test)
    for i in range(end-100, end):
        tmp_id = docid_list[i]
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            tmp_v = [1.0]
            tmp_v.extend(list(vector))
            x_test.append(tmp_v)
        else:
            continue    
    y_test = [1] * n_pos
    y_test.extend([0] * (len(x_test) - n_pos))

    return x_train, y_train, x_test, y_test


# Logistic Regression算法相关函数
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
    print("up_times:" + str(cycle_number))
    return theta

if __name__ == '__main__':
    # 预处理数据
    print("\n... 1.preprocess data ...\n")
    start_t = time.time()
    
    all_data = pd.read_table("./Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",
                        delimiter=' ', header=None, names=["QID", "x0", "DocID",
                        "score", "relevancy", "x1"])
    sentence_dict = preprocess()
    
    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We preprocess all data successfully!")
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")

    # Logistic Regression算法
    print("\n... 2.Logistic Regression algorith ...\n")
    start_t = time.time()
    
    for xx in range(201, 211):
        x_train, y_train, x_test, y_test = splitDataSet(xx, all_data,
                                                    sentence_dict)
        T = gradAscent(x_train, y_train)
        
        # 预测结果
        tmp_results = sigmoid(np.mat(x_test) * T)
        predict_results = []
        for res in tmp_results:
            if res > 0.5:
                predict_results.append(1)
            else:
                predict_results.append(0)
        # 计算准确率
        error_number = 0
        for i in range(len(predict_results)):
            if predict_results[i] != y_test[i]:
                error_number += 1
        accuracy = (1 - error_number / len(predict_results)) * 100
        print("QID:" + str(xx) + " Accuracy:" + str(accuracy) + '%')	

    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")
    print("\n... end, happy to see you ...\n")
