import time
import argparse

import _dynet as dy

import models
from lib import *


def test(parser, test_data, evalb_dir):
    test_predicted = []
    rvnn_scores_diff = []
    for data in test_data:
        _, predicted, rvnn_score_diff = parser.parse(data, is_train=False)
        test_predicted.append(predicted)
        rvnn_scores_diff.append(rvnn_score_diff)

    test_trees = [data['tree'] for data in test_data]
    test_fscore = evaluate.evalb(evalb_dir, test_trees, test_predicted)
    return test_fscore, rvnn_scores_diff


if __name__ == '__main__':
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

    model = dy.ParameterCollection()
    [parser] = dy.load(config.load_model_path + args.model + "_model", model)
    print("Loaded model from {}".format(
        config.load_model_path + args.model + "_model"))

    testing_data = parser.vocab.gold_data_from_file(config.test_file)
    print("Loaded testing data from {}".format(config.test_file))

    start_time = time.time()
    test_fscore, _ = test(parser, testing_data, config.evalb_dir)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
