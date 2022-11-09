import re
import pandas as pd
import numpy as np
from numpy import transpose

# 预处理数据相关函数
 
# 分词,去除停用词,返回(每个邮件内容的list,对应的标签的list),tag=0分训练集,tag=1分测试集
def preprocess():
    """ split all sentences and drop out stopwords """
    sentence_dict = {}
    # in the dictionary, key is a docID, value is a list containing all words
    # in a sentence(title+body)
    # by using UTF-8 we can get all stopwords
    swfile = open("./stopwords.txt", 'r', encoding='UTF-8')
    stopwords = swfile.read().splitlines()
    # we must drop out stopwords at the same time to save time
    line_gen = readLine("./documents.txt")
    # use a generator to read the huge file
    for line in line_gen:
        tmp_re = re.search(r'<article_id>(.*)</article_id>.*<title>(.*)</title>.*<body>(.*)</body>', line, flags=0)  # who can teach me how to wrap this line!!!!!!!!!!!!!!!
        if tmp_re is not None:
            docID = tmp_re.group(1)
            title = tmp_re.group(2)
            body = tmp_re.group(3)
            if not body.startswith(' '+title):
                body = title + body
            tmp_body = body.split()
            # drop out stopwords
            for i in range(len(tmp_body)-1, -1, -1):
                if ((len(tmp_body[i])<3) or (tmp_body[i] in stopwords)):
                # we consider 1 letter or 2 letter words useless
                    tmp_body.pop(i)
            sentence_dict[docID] = tmp_body
    print(len(sentence_dict))
    return sentence_dict

def readLine(file):
    """ use a generator to read a very big file """
    with open(file, 'r', encoding='UTF-8') as f:
        n = 0
        while True:
        # if you use 32-bit python interpreter, code this: while n < 6100000
            n += 1
            print(n)  # show the progress
            l = f.readline()
            if l:
                yield l
            else:
                return

# get one-hot word vector functions
def getAllWordsList(tmpquery_list):
    """ help to get one-hot representation of word vectors. you should
        respectively call it 10 times when you use different queries"""    
    wordsset = []
    for wl in tmpquery_list:
        for wd in wl:
            wordsset.append(wd)
    # set() can help delete repeated words
    allwords_list = list(set(wordsset))
    return allwords_list

def splitDataSet(QID, all_data, sentence_dict):
    """ standardize data and divide them into x_train y_train x_test y_test """
    start = (QID - 201) * 10000
    end = start + 10000
    docid_list = list(all_data['DocID'])
    # build one-hot presentation of word vector
    tmpquery_list = []
    for did in docid_list[start:end]:
        if sentence_dict.get(did) is not None:
            tmpquery_list.append(sentence_dict[did])
    allwords_list = getAllWordsList(tmpquery_list)
    
    # get trainset
    x_train = []
    for i in range(start+100, start+300):
        tmp_did = docid_list[i]
        # we ignore those examples which can't be found corresponding text
        if sentence_dict.get(tmp_did) is not None:
            wl = sentence_dict[tmp_did]
            tmp_v = [1.0]
            v = [0.0 for x in range(len(allwords_list))]
            for wd in wl:
                v[allwords_list.index(wd)] += 1.0
            tmp_v.extend(v)
            x_train.append(tmp_v)
        else:
            continue
    n_pos = len(x_train)
    for i in range(end-300, end-100):
        tmp_did = docid_list[i]
        # we ignore those examples which can't be found corresponding text
        if sentence_dict.get(tmp_did) is not None:
            wl = sentence_dict[tmp_did]
            tmp_v = [1.0]
            v = [0.0 for x in range(len(allwords_list))]
            for wd in wl:
                v[allwords_list.index(wd)] += 1.0
            tmp_v.extend(v)
            x_train.append(tmp_v)
        else:
            continue
    y_train = [1] * n_pos
    y_train.extend([0] * (len(x_train) - n_pos))

    # get testset
    x_test = []
    for i in range(start, start+100):
        tmp_did = docid_list[i]
        # we ignore those examples which can't be found corresponding text
        if sentence_dict.get(tmp_did) is not None:
            wl = sentence_dict[tmp_did]
            tmp_v = [1.0]
            v = [0.0 for x in range(len(allwords_list))]
            for wd in wl:
                v[allwords_list.index(wd)] += 1.0
            tmp_v.extend(v)
            x_test.append(tmp_v)
        else:
            continue
    n_pos = len(x_test)
    for i in range(end-100, end):
        tmp_did = docid_list[i]
        # we ignore those examples which can't be found corresponding text
        if sentence_dict.get(tmp_did) is not None:
            wl = sentence_dict[tmp_did]
            tmp_v = [1.0]
            v = [0.0 for x in range(len(allwords_list))]
            for wd in wl:
                v[allwords_list.index(wd)] += 1.0
            tmp_v.extend(v)
            x_test.append(tmp_v)
        else:
            continue
    y_test = [1] * n_pos
    y_test.extend([0] * (len(x_test) - n_pos))

    return x_train, y_train, x_test, y_test

# Logistic Regression algorith functions
def sigmoid(inX):
    """ sigmoid function """
    return 1.0 / (1.0 + np.exp(-inX))

def gradAscent(dataList, labelList):
    """ use gradient descent algorithm to get theta(vector) """
    # convert to NumPy matrix
    dataMatrix = np.mat(dataList)  # ..*len(wordvector)+1
    labelMatrix = np.mat(labelList).transpose()  # ..*1
    m, n = np.shape(dataMatrix)  # col - [1.0, .., ..]
    alpha = 0.1  # learning rate which can be customize
    theta = np.random.rand(n, 1)#np.ones((n,1))
    # ..*1: theta is a vector which you want to optimize

    cycle_number = 0  # loop number
    gra_abs = 1
    while gra_abs > 0.001 and cycle_number < 100000:
        sv = sigmoid(dataMatrix * theta)
        # ..*1: all examples' sigmoid value
        error = sv - labelMatrix
        # ..*1: sigmoid value - label
        gradient = dataMatrix.transpose() * error
        # len(wordvector)+1*1: gradient
        theta = theta - alpha * gradient
        gra_abs = min(gradient)
        cycle_number += 1
    return theta

if __name__ == '__main__':
    # preprocess data
    print("\n... 1.preprocess data ...\n")
    all_data = pd.read_table("./Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",
                        delimiter=' ', header=None, names=["QID", "x0", "DocID",
                        "score", "relevancy", "x1"])
    sentence_dict = preprocess()
    print("We preprocess all data successfully")

    # Logistic Regression algorith
    print("\n... 2.Logistic Regression algorith ...\n")
    # get one-hot word vector used in trainset and testset
    for xx in range(201,211):
        x_train, y_train, x_test, y_test = splitDataSet(xx, all_data,
                                                    sentence_dict)
        T = gradAscent(x_train, y_train)
        # get predict_results
        tmp_results = sigmoid(np.mat(x_test) * T)
        predict_results = []
        for res in tmp_results:
            if res > 0.5:
                predict_results.append(1)
            else:
                predict_results.append(0)
        # calculate accuracy
        error_number = 0
        for i in range(len(predict_results)):
            if predict_results[i] != y_test[i]:
                error_number += 1
        accuracy = (1 - error_number / len(predict_results)) * 100
        print("QID:" + str(xx) + " Accuracy:" + str(accuracy) + '%')	

    print("\n... end, happy to see you ...\n")
