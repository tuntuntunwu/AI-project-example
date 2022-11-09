# Constituent Parsing

The Parser uses chart-based model in stern et al. which don't need you to implement. The only thing you will do is using the RvNN to calculate the scores of the predicted tree and gold trees. Then calculate the differences of the two scores.

# Requirements and Setup

* Linux.
* Python 3.5 or higher.
* [Dynet]. Use command `pip install dynet` to install Dynet.
* Before running the code, first run the command `chmod 777 run/EVALB/evalb` to ensure that you have the permission to call the lib.

# Train
`python3 run/train.py`

# Test
`python3 run/test.py --dev_file data/dev.txt`

# To Do

Two function `get_parent_encoding()` and `get_tree_score()` in `run/models/RvNNParser.py`.  Refer to the comments in the code for more information.

**DON'T modify any other code of the project!**

**HAND IN:**

* Two model files in `ckpt/RvNNParser/`, named `RvNNParser_model.data` and `RvNNParser_model.meta`.
* A result file in `ckpt/RvNNParser/`, named `results.txt` .
* Code file `RvNNParser.py`.
* Project report (explain the process that you setup the environment and run the project, etc.).

# Experiments

My final results of the project is as following:

* F1: 93.18
* Zero count: 85