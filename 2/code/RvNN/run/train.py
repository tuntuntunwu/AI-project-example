import argparse
import sys
sys.setrecursionlimit(10000)
import time
import os
import itertools

import _dynet as dy
import numpy as np

import models
from lib import *
from test import *


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='configs/RvNNParser.cfg')
    argparser.add_argument(
        '--model', default='RvNNParser', choices=['RvNNParser'])
    args, extra_args = argparser.parse_known_args()
    args.config_file = "configs/{}.cfg".format(args.model)
    config = Configurable(args.config_file, extra_args)

    dyparams = dy.DynetParams()
    dyparams.set_autobatch(config.dynet_autobatch)
    dyparams.set_random_seed(config.dynet_seed)
    dyparams.set_mem(config.dynet_mem)
    dyparams.init()

    np.random.seed(config.numpy_seed)

    vocab = Vocabulary(config.train_file)
    training_data = vocab.gold_data_from_file(config.train_file)
    print('Loaded {} training sentences!'.format(len(training_data)))
    deving_data = vocab.gold_data_from_file(config.dev_file)
    print('Loaded {} validation sentences!'.format(len(deving_data)))

    model = dy.ParameterCollection()
    trainer = dy.AdamTrainer(model)

    Parser = getattr(models, args.model)
    parser = Parser(
        model,
        vocab,
        config.word_embedding_dim,
        config.tag_embedding_dim,
        config.lstm_layers,
        config.lstm_dim,
        config.fc_hidden_dim,
        config.dropout,
        config.unk_param,
    )

    current_processed = 0
    check_every = len(training_data) / config.checks_per_epoch
    best_dev_fscore = FScore(-np.inf, -np.inf, -np.inf)
    best_dev_epoch = 0
    best_dev_model_path = "{}{}_model".format(
        config.save_model_path, args.model)

    start_time = time.time()

    for epoch in itertools.count(start=1):
        if config.epochs is not None and epoch > config.epochs:
            break

        print('........... epoch {} ...........'.format(epoch))
        np.random.shuffle(training_data)
        total_loss_count = 0
        total_loss_value = 0.0

        for start_index in range(0, len(training_data), config.batch_size):
            batch_losses = []
            dy.renew_cg()

            for data in training_data[start_index:start_index + config.batch_size]:
                loss, _, _ = parser.parse(data)
                batch_losses.append(loss)
                current_processed += 1
                total_loss_count += 1

            batch_loss = dy.esum(batch_losses)
            total_loss_value += batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "\r"
                "batch {:,}/{:,}  "
                "mean-loss {:.4f}  "
                "total-elapsed {}  ".format(
                    start_index // config.batch_size + 1,
                    int(np.ceil(len(training_data) / config.batch_size)),
                    total_loss_value / total_loss_count,
                    format_elapsed(start_time)
                ),
                end=""
            )
            sys.stdout.flush()

            if current_processed >= check_every:
                current_processed -= check_every
                dev_fscore, rvnn_scores_diff = test(
                    parser, deving_data, config.evalb_dir)
                print("[Dev: {}]".format(dev_fscore))
                if dev_fscore.fscore > best_dev_fscore.fscore:
                    for ext in [".data", ".meta"]:
                        path = best_dev_model_path + ext
                        if os.path.exists(path):
                            os.remove(path)

                    best_dev_fscore = dev_fscore
                    best_dev_epoch = epoch
                    print("    [Saving new best model to {}]".format(
                        best_dev_model_path))
                    dy.save(best_dev_model_path, [parser])

                    fh = open(config.save_model_path + 'results.txt', 'w')
                    rvnn_scores_diff_sum = 0
                    zeros_cnt = 0
                    for rvnn_score_diff in rvnn_scores_diff:
                        fh.write(str(abs(rvnn_score_diff.value())) + '\n')
                        rvnn_scores_diff_sum += abs(rvnn_score_diff.value())
                        if rvnn_score_diff.value() == 0:
                            zeros_cnt += 1
                    fh.write("Average:\n")
                    fh.write(str(rvnn_scores_diff_sum / len(deving_data)) + '\n')
                    fh.write("Zero count:\n")
                    fh.write(str(zeros_cnt) + '/' +
                             str(len(deving_data)) + '\n')
                    fh.close()
                    print(
                        "    [Zero count: {}/{}]".format(zeros_cnt, len(deving_data)))

                    print("    [Saving new best RvNN scores differences to {}]".format(
                        config.save_model_path + 'results.txt'))

        print("[Best dev: {}, best epoch {}]".format(
            best_dev_fscore, best_dev_epoch))
