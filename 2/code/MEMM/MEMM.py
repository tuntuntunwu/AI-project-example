import math

class MaxEnt:
	def __init__(self, threshold=1e-2, max_iter=1000):
		self.samples = []  # 样本集, 元素是[y,x1,x2,...,xn]的列表
		self.X = []  # 样本输入集，元素是[X1,X2,...,XN]
		self.Y = set([])  # 标签集合，相当于去重之后的y
		self.num_Y = 0  # 类别个数
		self.N = 0  # 样本数量
		self.n = 0  # 特征对(xi,y)总数量
		self.xy_list = []  # 保存全部(xi,y)的列表
		self.M = 0  # 样本最大的特征数量，用于求参数时的迭代，详见IIS原理说明
		self.sample_ep = []  # 特征函数关于样本分布的期望
		self.model_ep = []  # 特征函数关于模型分布的期望
		self.W = []  # 对应n个特征的权值
		self.last_W = []  # 上一轮迭代的权值
		self.threshold = threshold  # 判断是否收敛的阈值
		self.max_iter = max_iter  # 最大迭代次数
	
	# 加载数据
	def load_data(self, filename):
		with open(filename, "r") as f:
			lines = f.read().strip().split("\n")
		xy_set = set()
		for line in lines:
			sample = line.split("\t")
			if len(sample) < 2:  # 至少：标签 + 一个特征
				continue
			y = sample[0]
			x = sample[1:]
			self.X.append(tuple(x))  # 样本输入
			self.Y.add(y)  # 样本标签
			self.samples.append(tuple(sample))  # 标签 + 特征
			if len(x) > self.M:  # 寻找样本最大的特征数量
				self.M = len(x)
			for xi in x:
				xy_set.add((xi, y))
		self.xy_list = list(xy_set)
	
	# 初始化模型参数
	def init_params(self):
		self.N = len(self.samples)
		self.n = len(self.xy_list)
		self.num_Y = len(self.Y)
		self.W = [0.0] * self.n
		self.last_W = self.W[:]
		self.calcu_sample_ep()
	
	# 计算特征函数关于样本分布的期望
	def calcu_sample_ep(self):
		self.sample_ep = [0.0] * self.n
		for sample in set(self.samples):
			x = sample[1:]
			y = sample[0]
			pxy = (self.samples.count(sample) * 1.0) / self.N  # 计算p(x,y)的经验分布
			for xi in x:
				index = self.xy_list.index((xi, y))
				self.sample_ep[index] += pxy
	
	# 模型训练
	def train(self):
		self.init_params()
		for i in range(self.max_iter):
			self.last_W = self.W[:]  # 保存上一轮权值
			self.calcu_model_ep()
			# 更新每个特征的权值
			for i, w in enumerate(self.W):
				self.W[i] += (1.0 / self.M) * math.log(self.sample_ep[i] / self.model_ep[i])
			# 检查是否收敛
			if self.judge_convergence():
				break
	
	# 计算特征函数关于模型分布的期望
	def calcu_model_ep(self):
		self.model_ep = [0.0] * self.n
		for sample in set(self.samples):
			x = sample[1:]
			y = sample[0]
			px = (self.X.count(x) * 1.0) / len(self.X)  # 计算p(X)的经验分布
			pyx_dict = self.clacu_pyx(x)  # 计算p(y|x)
			pyx = pyx_dict[y]
			for xi in x:
				index = self.xy_list.index((xi, y))
				self.model_ep[index] += px * pyx
	
	# 计算条件概率p(y|x)
	def clacu_pyx(self, x):
		Zw = 0.0  # 规范化因子
		pyx_temp = {}  # 保存pyx的分子
		pyx_dict = {}  # 保存pyx
		for y in self.Y:
			sum = 0.0
			for xi in x:
				if (xi, y) in self.xy_list:  # 这个判断相当于指示函数的作用
					index = self.xy_list.index((xi, y))
					sum += self.W[index]
			temp = math.exp(sum)
			pyx_temp[y] = temp
			Zw += temp
		for y in self.Y:
			pyx_dict[y] = pyx_temp[y] / Zw
		return pyx_dict
	
	# 根据阈值判断权值是否收敛
	def judge_convergence(self):
		for w, last_w in zip(self.W, self.last_W):
			change = math.fabs(w - last_w)
			if change >= self.threshold:
				return False
		return True
	
	# 预测新样本类别
	def predict(self, input):
		X = input.strip().split("\t")
		prob = self.clacu_pyx(X)
		return prob


if __name__ == "__main__":
	maxent = MaxEnt(threshold=1e-2, max_iter=1000)
	maxent.load_data('./data.txt')
	# 训练
	maxent.train()
	# 预测，自己设置预测用例
	print("rainy hot normal TRUE:")
	print(maxent.predict("rainy\thot\tnormal\tTRUE"))
	print("overcast mild high TRUE:")
	print(maxent.predict("overcast\tmild\thigh\tTRUE"))
	print("overcast hot high FALSE:")
	print(maxent.predict("overcast\thot\thigh\tFALSE"))
	print("sunny mild normal FALSE:")
	print(maxent.predict("sunny\tmild\tnormal\tFALSE"))
	print("rainy cool high TRUE:")
	print(maxent.predict("rainy\tcool\thigh\tTRUE"))
