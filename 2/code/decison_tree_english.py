from sklearn import tree
from sklearn.model_selection import train_test_split

import pandas as pd
import re

# preprocess data functions - spilt sentences + drop out stopwords
def preprocess(all_data):
	""" split all sentences and drop out stopwords """
	
	sentence_list = []
	# in the list, each element is a list containing its words
	for s in all_data['content']:
		# don't use _(underline) to split since they are complete words
		word_list = s.split()
		sentence_list.append(word_list)
	
	# by using UTF-8 we can get all stopwords
	swfile = open("./stopwords.txt", 'r', encoding='UTF-8')
	stopwords = swfile.read().splitlines()
	# drop out stopwords
	for wl in sentence_list:
		# TODO
		for i in range(len(wl)-1,-1,-1):
			# TODO if ((len(wl[i])<3) or (wl[i] in stopwords)):
			if ((len(wl[i])<3) or (wl[i] in stopwords)):
				wl.pop(i)
	
	return sentence_list

# get one-hot word vector functions
def getAllWordsList(sentence_list):
	""" help to get one-hot representation of word vector """
	
	wordsset = []
	for wl in sentence_list:
		for wd in wl:
			wordsset.append(wd)
	# set() can help you delete repeated words
	allwords_list = list(set(wordsset))
	return allwords_list


data = {}
# store 1.(e1,e2) 2.featurevector and 3.label
def getE1E2(all_data):
	""" get pairs of entities - a tuple (e1, e2) """

	P = []
	for i in range(len(all_data)):
		P.append((all_data['e1'][i], all_data['e2'][i]))
	data['e1e2'] = P[:]

def getFeatureVector(all_data, sentence_list, allwords_list):
	""" get standard data feature vector - each sentence's one-hot word vector,
		we view every word in word vector as a feature """

	X = []
	for i in range(len(sentence_list)):
		v = [0]*len(allwords_list)
		for wd in sentence_list[i]:
			v[allwords_list.index(wd)] += 1
		X.append(v)
	data['feature'] = X[:]

def getLabel(all_data):
	""" get standard data label - 0:person/place_lived 1:person/nationality
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

# decision tree algorithm function


if __name__ == '__main__':
	# preprocess data
	print("\n... 1.preprocess data ...\n")
	all_data = pd.read_excel("./train_ultimate.xlsx", header=None,
							names=["e1", "e2", "relation","content"])
	sentence_list = preprocess(all_data)
	allwords_list = getAllWordsList(sentence_list)
	getE1E2(all_data)
	getFeatureVector(all_data, sentence_list, allwords_list)
	getLabel(all_data)
	print("We input all samples successfully")

	# decision tree algorithm
	print("\n... 2.decision tree algorithm ...\n")
	x_train, x_test, y_train, y_test = train_test_split(data['feature'],
												data['label'], test_size = 0.2)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
	y_predict = clf.predict(x_test)

	# calculate accuracy
	error_number = 0
	for i in range(len(y_predict)):
		if y_predict[i] != y_test[i]:
			error_number += 1
	error_rate = 100 * error_number / len(y_predict)
	print("error_rate: " + str(error_rate) + '%')
	# calculate RMSE
	# when the predicted one is the same with the real one:
	# we think y-y^=0
	# when the predicted one isn't equal to the real one:
	# we think y-y^=1
	

	print("\n... end, happy to see you ...\n")
