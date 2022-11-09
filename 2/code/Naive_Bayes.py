from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import numpy as np
import os

# 预处理相关函数

# 分词,去除停用词,返回(每个邮件内容的list,对应的标签的list),tag=0分训练集,tag=1分测试集
def preprocess(mail_list, tag):
    """ split all sentences and drop out stopwords """
    label_list = []
    sentence_list = []

    # 非常感谢数据集给的这么好！！!
    # 非常感谢数据集给的这么好！！!
    # 非常感谢数据集给的这么好！！!
    # 分词
    if tag == 0:
        for mail in mail_list:
            label_list.append(int(mail[0]))
            sentence_list.append(mail[2:].split())
    else:
        for mail in mail_list:
            sentence_list.append(mail.split())
    # sentence_list形式为[[...], [...], ...] 每个小list里都是一个邮件内容的所有单词
    # label_list存放每个邮件的标签

    # 去除停用词
    with open("./stopwords.txt", encoding='UTF-8') as swfile:
        stopwords = swfile.read().splitlines()
    for sentence in sentence_list:
        # 只包含1个字母和2个字母的单词没什么用,可以去除
        for i in range(len(sentence)-1,-1,-1):
            if ((len(sentence[i])<3) or (sentence[i] in stopwords)):
                sentence.pop(i)
    
    return sentence_list, label_list


# 生成向量相关函数

# 生成每个邮件内容的文档向量(使用Doc2Vec生成)
def getDoc2Vec(sentence_list):
    """ use gensim.models.doc2vec - train word2vec, train and get doc2vec """
    # 这是官方示例代码,改了我就凉了
    documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(sentence_list)]
    # 词频小于min_count的词将被忽略,workers=8指在得到文档向量时使用8线程并行,训练起来会快
    model = Doc2Vec(documents, vector_size=10, min_count=1, workers=8)
    return model


# 朴素贝叶斯相关函数
# 本算法根据周志华西瓜书实现
# 以 文档向量的每一维值 作为特征进行朴素贝叶斯分类

# 建立贝叶斯分类器 - 即得到 在贝叶斯分类(predict)时所需的所有参数
def bulidBayesClassifier(label_list, model):
    """ build Bayes Classifier - get all parameters used for testset """
    # 得到类先验概率P(c),P[0]为类0的先验概率P(c0),P[1]为类1的先验概率P(c1)
    P = []
    
    n = 0
    for l in label_list:
        if l == 0:
            n += 1
    P0 = n / len(label_list)
    P1 = 1 - P0
    P.append(P0)
    P.append(P1)

    # 文档向量的每一维都为连续属性,所以应该使用概率密度函数计算后验概率p(xi|c)
    # 其应该服从正态分布,需要计算均值与方差

    # 得到均值-方差表,形式为[[(a0,v0),(a1,v1)],[...],...],表长为文档向量维数(特征数)
    # 表中每个元素是一个包含2个元组的列表,第一个元组是类0下此维(特征)的后验概率p(xi|c0)服从
    # 的分布的均值与方差,第二个元组是类1下此维(特征)的后验概率p(xi|c1)服从的分布的均值与方差
    a_v_list = []
    
    for i in range(len(model[0])):  # len(model[.]) == 100
        tmp_list0 = []
        tmp_list1 = []
        for j in range(len(label_list)):  # len(model) == 5000
            if label_list[j] == 0:
                tmp_list0.append(model[j][i])
            else:
                tmp_list1.append(model[j][i])
        a0 = np.average(tmp_list0)
        v0 = np.var(tmp_list0)
        a1 = np.average(tmp_list1)
        v1 = np.var(tmp_list1)
        lt = []
        lt.append((a0,v0))
        lt.append((a1,v1))
        a_v_list.append(lt)
    
    return P, a_v_list

# 求相应概率密度,因为是连续属性，因此不需要进行拉普拉斯修正
def pi(value, ttuple):
    """ p = 1/√2πσ^2 * e^(-(x-μ)^2/2σ^2) """
    p = 1 / np.sqrt(2 * np.pi * ttuple[1])
    p *= np.exp(- (value-ttuple[0])**2 / (2*ttuple[1]))
    return p

# 利用分类器进行分类(predict),得出结果
def predict(sentence_list, model, P, a_v_list):
    """ use Bayes Classifier to predict """
    predict_results = []
    # 对每个邮件生成文档向量,并预测
    for sentence in sentence_list:
        vector = model.infer_vector(sentence)
        pp0 = P[0]
        for i in range(len(vector)):
            pp0 *= pi(vector[i], a_v_list[i][0])
        pp1 = P[1]
        for i in range(len(vector)):
            pp1 *= pi(vector[i], a_v_list[i][1])
        
        if pp0 > pp1:
            predict_results.append(0)
        else:
            predict_results.append(1)
    
    return predict_results


if __name__ == '__main__':
    
    # 预处理数据
    print("\n... 1.preprocess data ...\n")
    start_t = time.time()
    
    with open("./spam_train.txt", encoding='UTF-8') as f:
        mail_list = f.readlines()
    sentence_list, label_list = preprocess(mail_list, 0)
    
    end_t = time.time()
    print("We preprocess all data successfully!")
    print("We use " + str(end_t-start_t) + " s")

    
    # 生成文档向量
    print("\n... 2.get mails' docvector ...\n")
    start_t = time.time()

    # 正确安装依赖可使gensim使用worker=4参数快速得到model
    # 如果无法并行处理,这一步可能将花费几小时的漫长时间,这时请使用已保存的模型
    if os.path.exists("./d2v.model"):
        model = Word2Vec.load("./d2v.model")
    else:
        model = getDoc2Vec(sentence_list)
        model.save("./d2v.model")
    # 设置不再修改model
    model.delete_temporary_training_data(keep_doctags_vectors=True,
                                        keep_inference=True)
    end_t = time.time()
    print("We get all mails' document vector!")
    print("We use " + str(end_t-start_t) + " s")

    
    # 朴素贝叶斯算法
    print("\n... 3.Naive Bayes algorith ...\n")
    start_t = time.time()
    
    # 建立贝叶斯分类器 - 即得到 在贝叶斯分类(predict)时所需的所有东西
    P, a_v_list = bulidBayesClassifier(label_list, model)
    
    end_t = time.time()
    print("We build Bayes Classifier successfully!")
    print("We use " + str(end_t-start_t) + " s")
    
    # 利用分类器对 训练集 进行分类(predict),得出训练集错误率以及TP、FN、FP、TN
    print("\nNow, we start to predict trainset...\n")
    
    predict_results = predict(sentence_list, model, P, a_v_list)
    # 错误率 与 TP、FN、FP、TN
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(predict_results)):
        if predict_results[i] == 0 and label_list[i] == 0:
            tp += 1
        elif predict_results[i] == 1 and label_list[i] == 0:
            fn += 1
        elif predict_results[i] == 0 and label_list[i] == 1:
            fp += 1
        elif predict_results[i] == 1 and label_list[i] == 1:
            tn += 1
    
    print("Trainset ErrorRate: " + str(100*(fn+fp)/len(predict_results)) + "%")
    print("TP: " + str(tp) + "\nFN: " + str(fn))
    print("FP: " + str(fp) + "\nTN: " + str(tn))
    
    # 利用分类器对 测试集 进行分类(predict),输出测试结果到predict.txt
    print("\nNow, we start to predict testset...\n")
    
    with open("./spam_test_no_label.txt", encoding='UTF-8') as f:
        mail_list = f.readlines()
    sentence_list, label_list = preprocess(mail_list, 1)
    predict_results = predict(sentence_list, model, P, a_v_list)
    with open("./predict.txt", 'w', encoding='UTF-8') as f:
        for r in predict_results:
            f.write(" " + str(r))
    
    print("We have written the predicted result for testset in 'predict.txt'!")

    print("\n... end, happy to see you ...\n")