import numpy as np
from numpy import transpose

# data
dataList = [
    [1.0, -0.017612, 14.053064],  # complement 1.0
    [1.0, -1.395634, 4.662541],
    [1.0, -0.752157, 6.538620],
    [1.0, -1.322371, 7.152853],
    [1.0, 0.423363, 11.054677],
    [1.0, 0.406704, 7.067335],
    [1.0, 0.667394, 12.741452],
    [1.0, -2.460150, 6.866805],
    [1.0, 0.569411, 9.548755],
    [1.0, -0.026632, 10.427743]
]
labelList = [
    0, 1, 0, 0, 0, 1, 0, 1, 0, 0
]

# PPT P4
# sigmoid function
def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))

# PPT P13
# use gradient descent algorithm to get theta(vector)
def gradDescent(dataList, labelList):
    # convert to NumPy matrix
    dataMatrix = np.mat(dataList)  # 10*3
    labelMatrix = np.mat(labelList).transpose()  # 10*1
    m, n = np.shape(dataMatrix)  # row:m = 10  col:n = 3 - [1.0, .., ..]
    alpha = 0.001  # learning rate which can be customize
    theta = np.ones((n,1))  # 3*1: theta is a vector which you want to optimize
    
    maxCycles = 1000  # loop number
    for i in range(maxCycles):
        sv = sigmoid(dataMatrix * theta)
        # 10*1: 10 examples' sigmoid value
        error = sv - labelMatrix  # 10*1: sigmoid value - label from 10 examples
        gradient = dataMatrix.transpose() * error  # 3*1: gradient
        theta = theta - alpha * gradient
    return theta

# test
def testLogisticRegression():
    T = gradDescent(dataList, labelList)
    predict_results = sigmoid(np.mat(dataList) * T)
    print("feature:\n", dataList, '\n')
    print("label:\n",labelList,'\n')
    print("theta:\n", T, '\n')
    print("predict_results:\n", predict_results)

if __name__ == '__main__':
    testLogisticRegression()
