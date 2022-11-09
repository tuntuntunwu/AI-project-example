import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

if __name__ == '__main__':
    
    # 数据读入
    print("\n... 1.read images ...\n")
    iris = pd.read_csv("./iris.data", header=None)
    iris_types = iris[4].unique()
    for i, type in enumerate(iris_types):
        iris.set_value(iris[4] == type, 4, i)
    data, label = np.split(iris.values, (4,), axis=1)

    # XGboost算法
    print("\n... 2.XGboost algorith ...\n")

    # 虽然搞不清楚怎样像调参网站一样使用cv函数，不过我们也可以完成调参任务
    # 使用循环或几个嵌套的循环可以直接实现调参的过程（para grid）
    # 简单更改不同变量取值范围可以实现对各个参数的调整
    bmd, bmcw, bac, bpr = 0, 0, 0.0, []
    mds = list(range(5,11,1))
    mcws = list(range(1,11,1))
    for md in mds:
        for mcw in mcws:
            X_train, X_test, Y_train, Y_test = train_test_split(data, label,
                                                test_size=0.2, random_state=0)
            dtrain = xgb.DMatrix(X_train, Y_train)
            dtest = xgb.DMatrix(X_test)
            
            # 设置XGboost训练参数
            params={
                # General Parameters
                'booster': 'gbtree',
                'silent': 1,
                #'nthread': 8,
        
                # Booster Parameters
                'eta': 0.025,  # learn rate
                #'min_child_weight': 3,  # used to control over-fitting
                'min_child_weight': mcw,  # used to control over-fitting
                #'max_depth': 6,  # used to control over-fitting
                'max_depth': md,  # used to control over-fitting
                #'max_leaf_nodes': 64,  # same to max_depth
                'gamma': 0.1,  # reduction in the loss function
                #'max_delta_step': 0  # be generally not used
                'subsample': 0.7,  # lower values prevents overfitting but too small lead to under-fitting.
                'colsample_bytree': 0.7,
                #'colsample_bylevel': 0.7,  # be generally not used
                'lambda': 2,  # L2 regularization term on weights, reduce over-fitting
                #'alpha': 0  # L1 regularization term on weights, run faster when high dimensionality
                #'scale_pos_weight': 1  # help in faster convergence when high class imbalance

                # Learning Task Parameters
                'objective': 'multi:softmax',
                'num_class': 3,
                #'eval_metric': 'auc',  # metric validation data
                'seed': 1000  # random number seed
                }
    
            # 训练、预测
            bst = xgb.train(params, dtrain, num_boost_round=500)
            predict_results = bst.predict(dtest)
            #xgb.to_graphviz(bst, num_trees=0)
    
            # 计算错误率
            error_number = 0
            m = len(Y_test)
            for i in range(m):
                if predict_results[i] != Y_test[i]:
                    error_number += 1
            accuracy = 100 * ((1 - error_number / m))
            print("(max_depth:%d, min_child_weight:%d)" %(md, mcw))
            print("Accuracy is:%f" %accuracy)
            # 存储最优参数
            if (accuracy > bac):
                bmd, bmcw, bac, bpr = md, mcw, accuracy, predict_results
    
    # 输出最优预测结果、错误率
    print("\nPredicted results:")
    print(bpr)
    with open("./predict.txt", 'w', encoding='UTF-8') as f:
        for predict_result in bpr:
            f.write(str(int(predict_result)) + ' ')    
    print("Accuracy is: %f" %(bac))

    print("\n... end, happy to see you ...\n")
