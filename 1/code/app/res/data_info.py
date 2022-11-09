import os
import json

class_number_dict = {}

root = "e:/Code/demo/res/单字"
dirs = os.listdir(root)
print(dirs)
with open("100+.json", 'w') as f:
    json.dump(dirs, f)