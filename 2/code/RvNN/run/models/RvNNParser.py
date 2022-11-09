import functools
import time

import _dynet as dy
import numpy as np

from lib import *


class RvNNParser(object):
    def network_init(self):
        self.tag_embeddings = self.model.add_lookup_parameters(
            (self.tag_count, self.tag_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.word_embeddings = self.model.add_lookup_parameters(
            (self.word_count, self.word_embedding_dim),
            init='uniform',
            scale=0.01
        )

        self.lstm = dy.BiRNNBuilder(
            self.lstm_layers,
            self.tag_embedding_dim + self.word_embedding_dim,
            2 * self.lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], self.label_out - 1)

        self.W = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.b = self.model.add_parameters((2 * self.lstm_dim,))

        self.U = self.model.add_parameters((1, 2 * self.lstm_dim))

    def __init__(
        self,
        model,
        vocab,
        word_embedding_dim,
        tag_embedding_dim,
        lstm_layers,
        lstm_dim,
        fc_hidden_dim,
        dropout,
        unk_param,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model
        self.vocab = vocab
        self.word_count = vocab.total_words()
        self.tag_count = vocab.total_tags()
        self.word_embedding_dim = word_embedding_dim
        self.tag_embedding_dim = tag_embedding_dim
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.label_out = vocab.total_label_actions()
        self.dropout = dropout
        self.unk_param = unk_param

        self.network_init()

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @staticmethod
    def augment(scores, oracle_index):
        assert isinstance(scores, dy.Expression)
        shape = scores.dim()[0]
        assert len(shape) == 1
        increment = np.ones(shape)
        increment[oracle_index] = 0
        return scores + dy.inputVector(increment)

    def get_embeddings(self, word_inds, tag_inds, is_train):
        embeddings = []
        for w, t in zip(word_inds, tag_inds):
            if w > 2:
                count = self.vocab.word_freq_list[w]
                if not count or (is_train and np.random.rand() < self.unk_param / (self.unk_param + count)):
                    w = 0

            tag_embedding = self.tag_embeddings[t]
            word_embedding = self.word_embeddings[w]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        return embeddings

    def parse(self, data, is_train=True):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right + 1][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 2][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(
                get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        @functools.lru_cache(maxsize=None)
        def get_parent_encoding(left_encoding, right_encoding):
            '''
            Input:
                Two  (vector representation) of left and right child.
            Output:
                Dynet expression of their parent.
                Use the following expression:
                    p = tanh(W * (l + r) + b)
                Tanh function can use dy.tanh().
                W and b have been defined in the class and use self.W or self.b to use them.
            '''

            # YOUR CODE HERE
            return dy.tanh(np.dot(self.W, (left_encoding + right_encoding)) + self.b)
            # END YOUR CODE

        @functools.lru_cache(maxsize=None)
        def get_tree_score(parent_encoding):
            '''
            Input:
                Parent Dynet expression.
            Output:
                Score of parent node.
                Use the following expression:
                    s = U * p
                U have been defined in the class and use self.U to use it.
            '''

            # YOUR CODE HERE
            return np.dot(self.U, parent_encoding)
            # END YOUR CODE

        def helper(force_gold):
            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length - 1

                    label_scores = get_label_scores(left, right)

                    oracle_label, crossing = gold_tree.span_labels(
                        left, right)
                    oracle_label = oracle_label[::-1]

                    if len(oracle_label) == 0:
                        oracle_label = 'none'
                    else:
                        oracle_label = 'label-' + '-'.join(oracle_label)
                    oracle_label_index = self.vocab.l_action_index(
                        oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                        label_index = oracle_label_index
                    else:
                        if is_train:
                            label_scores = RvNNParser.augment(
                                label_scores, oracle_label_index)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)

                        if argmax_label_index == 0:
                            argmax_label = 'none'
                        else:
                            argmax_label = 'label-' + \
                                self.vocab.i2l[argmax_label_index - 1]

                        label = argmax_label
                        label_score = label_scores[argmax_label_index]
                        label_index = argmax_label_index

                    if length == 1:
                        tree = PhraseTree(leaf=left)
                        if label != 'none':
                            for nt in label[6:].split('-'):
                                tree = PhraseTree(symbol=nt, children=[tree])
                        leaf_encoding = get_span_encoding(left, right)
                        rvnn_score = get_tree_score(leaf_encoding)

                        chart[left, right] = [
                            tree], label_score, leaf_encoding, rvnn_score
                        continue

                    if force_gold:
                        oracle_splits = gold_tree.span_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = left + 1
                        best_split_score = -np.inf
                        for split in range(left + 1, right + 1):
                            left_score = chart[left, split - 1][1].value()
                            right_score = chart[split, right][1].value()
                            split_score = left_score + right_score

                            if split_score > best_split_score:
                                best_split_score = split_score
                                best_split = split

                    left_trees, left_score, left_encoding, left_rvnn_score = \
                        chart[left, best_split - 1]
                    right_trees, right_score, right_encoding, right_rvnn_score = chart[
                        best_split, right]
                    parent_encoding = get_parent_encoding(
                        left_encoding, right_encoding)
                    rvnn_score = get_tree_score(parent_encoding)

                    childrens = left_trees + right_trees
                    if label != 'none':
                        for nt in label[6:].split('-'):
                            childrens = [PhraseTree(
                                symbol=nt, children=childrens)]

                    chart[left, right] = (
                        childrens, label_score + left_score + right_score, parent_encoding,
                        rvnn_score + left_rvnn_score + right_rvnn_score)

            childrens, score, encoding, rvnn_score = chart[0, len(
                sentence) - 1]
            assert len(childrens) == 1
            return childrens[0], score, encoding, rvnn_score

        tree, score, encoding, rvnn_score = helper(False)
        tree.propagate_sentence(sentence)

        oracle_tree, oracle_score, oracle_encoding, oracle_rvnn_score = helper(
            True)
        oracle_tree.propagate_sentence(sentence)

        assert str(oracle_tree) == str(gold_tree)

        if is_train:
            correct = (str(tree) == str(oracle_tree))
            loss = dy.zeros(1) if correct else score - oracle_score
            return loss, tree, rvnn_score - oracle_rvnn_score
        else:
            return score, tree, rvnn_score - oracle_rvnn_score
