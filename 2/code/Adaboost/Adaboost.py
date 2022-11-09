from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import re
import pandas as pd
import numpy as np
from numpy import transpose
import time

## 数据集处理

# 分词，去停用词，返回存储了 文章ID 与 对应内容 的字典sentence_dict
def preprocess():
    # sentence_dict字典存储 文章ID 与 其对应的文本内容 
    sentence_dict = {}
    
    # 通常的笔记本中，完整地运行数据处理代码需1个半小时左右
    # 因此我们将处理好的内容存放在sentence_dict.txt
    if os.path.exists("./sentence_dict.txt"):
        with open("./sentence_dict.txt", encoding='UTF-8') as f:
            for line in f.readlines():
                tmp = line.split()
                sentence_dict[tmp[0]] = tmp[1:]
    else:
        swfile = open("./stopwords.txt", encoding='UTF-8')
        stopwords = swfile.read().splitlines()
        
        #  使用生成器读取大文件
        line_gen = readFileByLine("./documents.txt")
        for line in line_gen:
            tmp_re = re.search(r'<article_id>(.*)</article_id>.*<title>(.*)</title>.*<body>(.*)</body>', line, flags=0)
            if tmp_re is not None:
                docID = tmp_re.group(1)
                title = tmp_re.group(2)
                body = tmp_re.group(3)
                if not body.startswith(' '+title):
                    body = title + body
                tmp_body = body.split()
            
                # 数据读取与去停用词同次进行，可节约时间
                for i in range(len(tmp_body)-1, -1, -1):
                    # 英文中小于3个字母的单词没有什么意义，被去除
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
def readFileByLine(file):
    with open(file, encoding='UTF-8') as f:
        n = 0
        # 如果你使用32位python解释器，将无法处理全部的file内容
        # 请这样写代码： while n < 6100000 来读取更可能多的内容
        while True:
            n += 1
            if n % 1000 == 0:
                print(n)
            l = f.readline()
            if l:
                yield l
            else:
                return


## 生成文章向量（特征）

# 生成能将文章内容转化为一个向量的模型(使用gensim.models模块中的Doc2Vec生成)
# 不同的query生成不同的模型
def getDoc2Vec(sentence_list):
    documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(sentence_list)]
    # 文章向量维数为100，训练生成模型时8线程并行
    model = Doc2Vec(documents, vector_size=100, min_count=1, workers=8)
    return model
# 划分数据集,返回对应query中需用到的 X_train Y_train X_test Y_test
def splitDataSet(QID, all_data, sentence_dict):
    start = (QID - 201) * 10000
    end = start + 10000
    docid_list = list(all_data['DocID'])
    # 生成能将文章内容变为文章向量的模型Doc2Vec
    query_sentence_list = []
    for _id in docid_list[start:end]:
        if sentence_dict.get(_id) is not None:
            query_sentence_list.append(sentence_dict[_id])
    model = getDoc2Vec(query_sentence_list)

    # X_train Y_train
    X_train, Y_train = [], []
    # 存在不能匹配到内容的文章，我们需多取一些数据才能得到近300个训练样本
    train_index = []
    train_index.extend(range(start + 100, start + 400))
    train_index.extend(range(end - 400, end - 100))
    for i in train_index:
        tmp_id = docid_list[i]
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            X_train.append(list(vector))
            if i < (start + 400):
                Y_train.append(1)
            else:
                Y_train.append(-1)

    # X_test Y_test
    X_test, Y_test = [], []
    test_index = []
    test_index.extend(range(start, start + 100))
    test_index.extend(range(end - 100, end))
    for i in test_index:
        tmp_id = docid_list[i]
        if sentence_dict.get(tmp_id) is not None:
            sentence = sentence_dict[tmp_id]
            vector = model.infer_vector(sentence)
            X_test.append(list(vector))
            if i < (start + 100):
                Y_test.append(1)
            else:
                Y_test.append(-1)

    return X_train, Y_train, X_test, Y_test


## Adaboost算法相关函数

# https://blog.csdn.net/buptgshengod/article/details/25049305
# 生成1个该条件下 最优的基学习器，返回它的参数，错误率，分类结果
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 鉴于之后的操作，我们将 初始错误率 设为+infinity
    numSteps = 10.0
    # 对每一维循环，找出最好的划分特征
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                
                threshVal = rangeMin + j * stepSize                 
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                
                # 计算带分布权重的错误率
                errArr = np.mat(np.ones((m, 1)))                
                errArr[predictedVals == labelMatrix] = 0
                weightedError = D.T * errArr
                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    return bestStump, minError, bestClasEst
# 决策树桩
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # lt: 大于
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    
    return retArray

# 预测
def predict(input_data, base_learners, learner_weights):
    dataMatrix = np.mat(input_data)
    
    predicted_results = np.mat(np.zeros((len(input_data), 1)))
    for i in range(len(base_learners)):
        base_learner = base_learners[i]
        predictedVals = stumpClassify(dataMatrix, base_learner['dim'],\
                            base_learner['thresh'], base_learner['ineq'])
        predicted_results += float(learner_weights[i]) * predictedVals
    
    predicted_results = np.sign(predicted_results)
    return predicted_results


if __name__ == '__main__':
    # 数据集处理
    print("\n... 1.preprocess data ...\n")
    
    all_data = pd.read_table("./Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",
                        delimiter=' ', header=None, names=["QID", "x0", "DocID",
                        "score", "relevancy", "x1"])
    sentence_dict = preprocess()
    
    print("We preprocess all data successfully!")

    # Adaboost算法
    print("\n... 2.Adaboost algorith ...\n")
    start_t = time.time()

    for xx in range(201, 211):
        print("QID: " + str(xx))
        X_train, Y_train, X_test, Y_test = splitDataSet(xx, all_data,\
                                                        sentence_dict)

        # 初始化数据分布
        m = len(Y_train)
        Dt = np.array([1.0 / m] * m)
        
        # 迭代
        loop_number = 400
        # 存储 生成的每个基学习器 和 其权重
        base_learners = []
        learner_weights = []
        for i in range(loop_number):
            
            # 得到 基学习器，错误率，该基学习器的分类结果预测 - 一个1/-1列表
            ht, et, Y_predict = buildStump(X_train, Y_train, Dt)
            print("et" + str(i) + " =" + str(et))
            if et > 0.5:
                print(">0.5 break")
                break
            base_learners.append(ht)
            
            # 计算该基学习器的 权重
            at = 0.5 * np.log((1 - et) / et)
            print("at" + str(i) + " =" + str(at))
            learner_weights.append(at)

            # 调整数据分布
            for i in range(m):
                if Y_predict[i] == Y_train[i]:
                    Dt[i] /= 2 * (1 - et)
                else:
                    Dt[i] /= 2 * et
        else:
            print("loop over break")
        
        # 预测
        # 训练集错误率
        predicted_results = predict(X_train, base_learners, learner_weights)
        error_number = 0
        m = len(Y_train)
        for i in range(m):
            if predicted_results[i] != Y_train[i]:
                error_number += 1
        print("train set's error rate: " + str(100 * error_number / m))
        # 测试集错误率
        predicted_results = predict(X_test, base_learners, learner_weights)
        error_number = 0
        m = len(Y_test)
        for i in range(m):
            if predicted_results[i] != Y_test[i]:
                error_number += 1
        print("test set's error rate: " + str(100 * error_number / m) + '\n')

        # 将测试集预测结果写入文件
        with open("predictQID" + str(xx) + ".txt", 'w', encoding='UTF-8') as f:
            for predicted_result in predicted_results:
                f.write(str(predicted_result) + '\n')
    
    end_t = time.time()
    m, s = divmod(end_t - start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")
    
    print("\n... end, happy to see you ...\n")
