import time

# 最基本的类，用来构建HMM的结构。根据不同的任务继承去不同子类具体实现
class HMModel:
    
    # 类中维护的模型参数均为频数而非频率，这样的设计使得模型可以进行在线学习
    # 在线学习包括：增量学习，减量学习
    def __init__(self):
        # trans_mat：状态转移矩阵
        # trans_mat[status1][status2] = int：训练集中由state1转移到state2的次数
        self.trans_mat = {}
        # emit_mat：观测矩阵
        # emit_mat[state][char] = int：训练集中单字char被标注为state的次数
        self.emit_mat = {}
        # init_vec：初始状态分布向量
        # init_vec[state] = int：训练集中状态state出现在状态序列首的次数
        self.init_vec = {}
        # state_count：状态统计向量，用于之后求概率
        # state_count[status] = int：训练集中状态state出现的总次数
        self.state_count = {}
        # states：状态集合
        self.states = {}
        self.inited = False
    
    # 始化HMM
    def setup(self):
        for state in self.states:
            # trans_mat
            self.trans_mat[state] = {}
            for target in self.states:
                self.trans_mat[state][target] = 0.0
            # emit_mat
            self.emit_mat[state] = {}
            # init_vec
            self.init_vec[state] = 0
            # state_count
            self.state_count[state] = 0
        self.inited = True

    # 加载数据
    def loadData(self, filename):
        self.data = open(filename, 'r', encoding="utf-8")
        self.setup()
    
    # 输入观测序列和状态序列进行训练, 依次更新各矩阵数据. 
    def doTrain(self, observes, states):
        if not self.inited:
            self.setup()

        for i in range(len(states)):
            # ？
            if i == 0:
                self.init_vec[states[0]] += 1
                self.state_count[states[0]] += 1
            else:
                self.trans_mat[states[i-1]][states[i]] += 1
                self.state_count[states[i]] += 1
                if observes[i] not in self.emit_mat[states[i]]:
                    self.emit_mat[states[i]][observes[i]] = 1
                else:
                    self.emit_mat[states[i]][observes[i]] += 1
    
    # 进行预测前需要将频数转换为频率
    def getProb(self):
        init_vec = {}
        trans_mat = {}
        emit_mat = {}
        # init_vec
        for key in self.init_vec:
            init_vec[key] = float(self.init_vec[key]) / self.state_count[key]
        # trans_mat
        for key1 in self.trans_mat:
            trans_mat[key1] = {}
            for key2 in self.trans_mat[key1]:
                trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / self.state_count[key1]
        # emit_mat
        for key1 in self.emit_mat:
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / self.state_count[key1]
        return init_vec, trans_mat, emit_mat
    
    # Viterbi算法预测
    def doPredict(self, sequence):
        # Viterbi表，记录各位置的各个状态的对应最大路径的概率
        tab = [{}]
        # 记录各候选最大路径的路径
        path = {}
        init_vec, trans_mat, emit_mat = self.getProb()

        # 初始化
        for state in self.states:
            tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0], EPS)
            path[state] = [state]

        # 动态搜索，建立Viterbi表，得到各候选最大路径
        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in self.states:
                items = []
                for state2 in self.states:
                    if tab[t-1][state2] == 0:
                        continue
                    prob = tab[t-1][state2] * trans_mat[state2].get(state1, EPS) * emit_mat[state1].get(sequence[t], EPS)
                    items.append((prob, state2))
                best = max(items)  # best: (prob, state)
                tab[t][state1] = best[0]
                new_path[state1] = path[best[1]] + [state1]
            path = new_path

        # 找出最大路径
        prob, state = max([(tab[len(sequence)-1][state], state) for state in self.states])
        return path[state]


STATES = {'B', 'M', 'E', 'S'}
EPS = 0.0001

def getTags(src):
    tags = []
    if len(src) == 1:
        tags = ['S']
    elif len(src) == 2:
        tags = ['B', 'E']
    else:
        m_num = len(src) - 2
        tags.append('B')
        tags.extend(['M'] * m_num)
        tags.append('S')
    return tags

def cutSent(src, tags):
    word_list = []
    start = -1
    started = False

    if len(tags) != len(src):
        return None

    if tags[-1] not in {'S', 'E'}:
        if tags[-2] in {'S', 'E'}:
            tags[-1] = 'S'  # for tags: r".*(S|E)(B|M)"
        else:
            tags[-1] = 'E'  # for tags: r".*(B|M)(B|M)"

    for i in range(len(tags)):
        if tags[i] == 'S':
            if started:
                started = False
                word_list.append(src[start:i])  # for tags: r"BM*S"
            word_list.append(src[i])
        elif tags[i] == 'B':
            if started:
                word_list.append(src[start:i])  # for tags: r"BM*B"
            start = i
            started = True
        elif tags[i] == 'E':
            started = False
            word = src[start:i+1]
            word_list.append(word)
        elif tags[i] == 'M':
            continue
    return word_list


# 继承HMModel，完成中文分词任务
class HMMSegger(HMModel):

    def __init__(self, *args, **kwargs):
        super(HMMSegger, self).__init__(*args, **kwargs)
        self.states = STATES
        self.data = None

    def train(self):
        # train
        for line in self.data.readlines():
            # pre processing
            line = line.strip()
            if not line:
                continue

            # get observes
            observes = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                observes.append(line[i])

            # get states
            words = line.split(" ")  # spilt word by whitespace
            states = []
            for word in words:
                #if word in seg_stop_words:
                    #continue
                states.extend(getTags(word))

            # resume train
            self.doTrain(observes, states)
    
    def cut(self, sentence):
        try:
            tags = self.doPredict(sentence)
            return cutSent(sentence, tags)
        except:
            return sentence


if __name__ == '__main__':
    start_t = time.time()

    # 训练HMMSegger
    segger = HMMSegger()
    segger.loadData("CTBtrainingset.txt")
    segger.train()
    
    # 使用HMMSegger预测
    results = []
    with open("CTBtestingset.txt", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            results.append(segger.cut(line))
    # 结果写入result.txt
    with open("result.txt", 'w', encoding="utf-8") as fw:
        for result in results:
            result = result[:-1]
            for segment in result:
                fw.write(segment)
                fw.write(" ")
            fw.write("\n")
    
    end_t = time.time()
    m, s = divmod(end_t-start_t, 60)
    print("We use " + str(round(m)) + " min " + str(round(s, 2)) + " s")