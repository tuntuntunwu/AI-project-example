import time
import numpy as np
import KNN

## PCA算法
def pca(data, k):
    # 标准化
    average = np.mean(data, axis=0)
    m, n = np.shape(data)
    data_norm = []
    avgs = np.tile(average, (m, 1))
    data_norm = data - avgs
    # 计算协方差矩阵
    covX = np.cov(data_norm.T)
    # 得出特征值、特征向量
    feat_val, feat_vec = np.linalg.eig(covX)
    # 按特征值大小排列，得出约简矩阵
    index = np.argsort(-feat_val)
    final_data = []
    select_vec = np.matrix(feat_vec.T[index[:k]])
    # 降维后的数据
    final_data = data_norm * select_vec.T
    # 还原到原始空间的数据
    recon_data = (final_data * select_vec) + average
    return final_data, recon_data


if __name__ == '__main__':
    
    # 数据读入
    print("\n... 1.read images and preprocess ...\n")
    train_img, train_label, test_img, test_label = KNN.getData()
    # 数据预处理
    train_img = KNN.biData(train_img)
    train_img = KNN.img2vec(train_img)
    test_img = KNN.biData(test_img)
    test_img = KNN.img2vec(test_img)
    print("We've read all images and preprocessed them!")

    start_t = time.time()
    
    # KNN
    print("\n... 2.only KNN algorith ...\n")
    # KNN
    predict_results = KNN.knn(3, train_img, train_label, test_img)
    # 准确率计算
    only_KNN_accuracy = KNN.calAccuracy(predict_results, test_label)
    print("Accuracy: " + str(only_KNN_accuracy) + "%")
    # 将结果写入"KNN_result.txt"文件
    with open("./only_KNN_result.txt", 'w', encoding='UTF-8') as f:
        for l in predict_results:
            f.write(str(l) + '\n')
    
    # KNN+PCA
    print("\n... 3.KNN+PCA algorith ...\n")
    # PCA
    k = 50
    x, train_final = pca(train_img, k)
    x, test_final = pca(test_img, k)
    # KNN
    predict_results = KNN.knn(3, train_final, train_label, test_final)
    # 准确率计算
    KNN_PCA_accuracy = KNN.calAccuracy(predict_results, test_label)
    print("Accuracy: " + str(KNN_PCA_accuracy) + "%")
    # 将结果写入"KNN+PCA_result.txt"文件
    with open("./KNN+PCA_result.txt", 'w', encoding='UTF-8') as f:
        for l in predict_results:
            f.write(str(l) + '\n')
    print("\nonly_KNN_accuracy: " + str(only_KNN_accuracy) + "%")
    print("KNN+PCA_accuracy: " + str(KNN_PCA_accuracy) + "%")
    
    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s\n")
    print("... end, happy to see you ...\n")
