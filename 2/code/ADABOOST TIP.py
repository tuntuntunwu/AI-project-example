from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import re
import pandas as pd
import numpy as np
from numpy import transpose
import time

X_train = [[1,5], [2,2], [3,1], [4,6], [6,8], [6,5], [7,9], [8,7], [9,8], [10,2]]
Y_train = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
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
    # Adaboost算法
    print("\n... 2.Adaboost algorith ...\n")
    start_t = time.time()

    m = len(Y_train)
    Dt = np.array([1.0 / m] * m)
        
    # 迭代
    loop_number = 8
    # 存储 生成的每个基学习器 和 其权重
    base_learners = []
    learner_weights = []
    for i in range(loop_number):
            
        # 得到 基学习器，错误率，该基学习器的分类结果预测 - 一个1/-1列表
        ht, et, Y_predict = buildStump(X_train, Y_train, Dt)
        print("et = " + str(et))
        if et > 0.5:
            print(">0.5 break")
            break
        base_learners.append(ht)
            
        # 计算该基学习器的 权重
        at = 0.5 * np.log((1 - et) / et)
        print("at = " + str(at))
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


    end_t = time.time()
    m, s = divmod(end_t - start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")
    
    print("\n... end, happy to see you ...\n")
