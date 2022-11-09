from sklearn import tree
from sklearn.model_selection import train_test_split

import pandas as pd
import re
import math

# 预处理数据的函数 - 划分句子 + 去除停用词
def preprocess(all_data):
	""" 划分所有的句子、去除停用词 """
	
	sentence_list = []
	# 这一列表中，每个元素都是一个包含一整个句子单词的列表
	for s in all_data['content']:
		word_list = s.split()
		sentence_list.append(word_list)
	
	# 使用splitlines()以及UTF-8，我们可以轻松得到所有的停用词
	swfile = open("./stopwords.txt", 'r', encoding='UTF-8')
	stopwords = swfile.read().splitlines()
	# 删除停用词
	for wl in sentence_list:
		for i in range(len(wl)-1,-1,-1):
			# 我们认为太过简短的，只包含1个单词或2个单词的词没什么用，可以去除
			if ((len(wl[i])<3) or (wl[i] in stopwords)):
				wl.pop(i)
	
	return sentence_list

# 帮助取得one-hot词向量的函数
def getAllWordsList(sentence_list):
	""" 返回有训练集中全部词的向量，他可以帮助我们之后生成one-hot词向量 """
	
	wordsset = []
	for wl in sentence_list:
		for wd in wl:
			wordsset.append(wd)
	# 用set()去除重复元素
	allwords_list = list(set(wordsset))
	return allwords_list


data = {}
# 我们使用一个字典来存储 1.(e1,e2) 2.特征向量 3.标签
def getE1E2(all_data):
	""" 存储实体对 - 每个元素是元组(e1, e2) """

	P = []
	for i in range(len(all_data)):
		P.append((all_data['e1'][i], all_data['e2'][i]))
	data['e1e2'] = P[:]

def getFeatureVector(all_data, sentence_list, allwords_list):
	""" 获得每个句子标准的one-hot词向量 - 我们将词向量中的每个词做为一个特征进行
		决策树算法 """

	X = []
	for i in range(len(sentence_list)):
		v = [0]*len(allwords_list)
		for wd in sentence_list[i]:
			v[allwords_list.index(wd)] += 1
		X.append(v)
	data['feature'] = X[:]

def getLabel(all_data):
	""" 获得标准的数据标签 - 0:person/place_lived 1:person/nationality
	    2:person/company 3:location/contains 4:NA """
	
	Y = []
	for rel in list(all_data['relation']):
		if re.search("place_lived", str(rel), flags=0):
			Y.append(0)
		elif re.search("nationality", str(rel), flags=0):
			Y.append(1)
		elif re.search("company", str(rel), flags=0):
			Y.append(2)
		elif re.search("contains", str(rel), flags=0):
			Y.append(3)
		else:
			Y.append(4)
	data['label'] = Y[:]


if __name__ == '__main__':
	# 预处理数据
	print("\n... 1.preprocess data ...\n")
	all_data = pd.read_excel("./train_ultimate.xlsx", header=None,
							names=["e1", "e2", "relation","content"])
	sentence_list = preprocess(all_data)
	# 获得one-hot词向量的过程
	allwords_list = getAllWordsList(sentence_list)
	getE1E2(all_data)
	getFeatureVector(all_data, sentence_list, allwords_list)
	getLabel(all_data)
	print("We input all samples successfully")

	# 决策树训练和测试
	print("\n... 2.decision tree algorithm ...\n")
	x_train, x_test, y_train, y_test = train_test_split(data['feature'],
												data['label'], test_size = 0.2)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
	y_predict = clf.predict(x_test)

	# 计算精准度
	error_number = 0
	for i in range(len(y_predict)):
		if y_predict[i] != y_test[i]:
			error_number += 1
	accuracy = 100 * (1 - error_number / len(y_predict))
	print("Accuracy: " + str(accuracy) + '%')
	
	# 计算 RMSE
	# 由于place_lived和nationality容易混淆，所以我们认为2者之间的错认交普遍， y-y^=0.5
	# 由于NA和其他类型相差甚远，如果被认成了NA，则y-y^=1.5
	# 其余错误均认为y-y^=1
	# 这样我们计算出的RMSE为：
	tmp = 0
	for i in range(len(y_predict)):
		if y_predict[i] != y_test[i]:
			if y_predict[i] == 4:
				tmp += 1.5**2
			elif y_predict[i] == 0 and y_test[i] == 1:
				tmp += 0.5**2
			elif y_predict[i] == 1 and y_test[i] == 0:
				tmp += 0.5**2
			else:
				tmp += 1**2
	rmse = math.sqrt(tmp / len(y_predict))
	print("RMSE: " + str(rmse))

	# 将预测的测试集关系输出到out.txt
	#all_data = pd.read_excel("./test.xlsx", header=None,
	#						names=["e1", "e2", "x1", "content", "x2"])
	#sentence_list = preprocess(all_data)
	# 获得one-hot词向量的过程
	#allwords_list = getAllWordsList(sentence_list)
	#getFeatureVector(all_data, sentence_list, allwords_list)
	#x_train, x_test, y_train, y_test = train_test_split(data['feature'],
	#											data['label'], test_size = 1)
	#clf = tree.DecisionTreeClassifier()
	#clf = clf.fit(x_train, y_train)
	#y_predict = clf.predict(x_test)
	with open("out.txt", 'w') as f:
		for i in y_predict:
			if i == 0:
				f.write("person/place_lived\n")
			elif i == 1:
				f.write("person/nationality\n")
			elif i == 2:
				f.write("person/company\n")
			elif i == 3:
				f.write("location/contains\n")
			else:
				f.write("NA\n")	

	print("\n... end, happy to see you ...\n")
