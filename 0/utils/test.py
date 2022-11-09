import os
import random
import shutil

import cv2
import numpy as np


from utils import *

# -----自己的crop和fill的strategy-----
# def huawei_ac_crop_strategy_1(img, instance):
#     size = 416
#     crop_strategies = []
#     width = instance['imageWidth']
#     height = instance['imageHeight']
#     top, bottom, left, right = get_edge(img)
#     crop_sizes = [[round(max([top-size/2, 0])), round(top+size/2), 0, width], [round(bottom-size/2), round(min([bottom+size/2, height])), 0, width], [0, height, round(max([0, left-size/2])), round(left+size/2)], [0, height, round(right-size/2), round(min([right+size/2, width]))]]
#     fill_sizes = [[round(max([size/2-top, 0])), 0, 0, 0], [0, round(max([bottom+size/2-height, 0])), 0, 0], [0, 0, round(max([size/2-left, 0])), 0], [0, 0, 0, round(max([0, right+size/2-width]))]]
#     for i, crop_size in enumerate(crop_sizes):
#         if not crop_is_empty(instance, crop_size):
#             crop_strategies.append([crop_size, fill_sizes[i]])
#     return crop_strategies
# def huawei_ac_crop_strategy_2(img, instance):
#     size = 416
#     overlap = 0.15
#     crop_strategies = []
#     width = instance['imageWidth']
#     height = instance['imageHeight']
#     num = get_crop_num(width, size, overlap) if height == size else get_crop_num(height, size, overlap)
#     for i in range(num):
#         if i == num-1:
#             crop_size = [0, height, width-size, width] if height == size else [height-size, height, 0, width]
#         else:
#             crop_size = [0, height, round(i*size*(1-overlap)), round(i*size*(1-overlap)+size)] if height == size else [round(i*size*(1-overlap)), round(i*size*(1-overlap)+size), 0, width]
#         if not crop_is_empty(instance, crop_size):
#             crop_strategies.append([crop_size, [0,0,0,0]])
#     return crop_strategies
# def huawei_side_crop_strategy_1(img, instance):
#     size = 768
#     overlap = 0.15
#     crop_strategies = []
#     width = instance['imageWidth']
#     min_x = int((width-size)/2)
#     height = instance['imageHeight']
#     num = get_crop_num(height, size, overlap)
#     for i in range(num):
#         if i == num-1:
#             crop_size = [height-size, height, min_x, min_x+size]
#         else:
#             crop_size = [round(i*size*(1-overlap)), round(i*size*(1-overlap))+size, min_x, min_x+size]
#         if not crop_is_empty(instance, crop_size):
#             crop_strategies.append([crop_size, [0,0,0,0]])
#     return crop_strategies
# def huawei_side_crop_strategy_2(img, instance):
#     pass
# def get_edge(img, thres=20):
#     '''
#     :param img: cv2读取的img对象
#     :param thres: 读到图片边缘像素的跳越阈值
#     :return: 图片中目标上下左右的坐标值
#     '''
#     result = []
#     h, w, c = img.shape
#     for i in range(h):
#         if img[i, int(w/2), 0] >= thres:
#             break
#     result.append(i)
#     for i in reversed(range(h)):
#         if img[i-1, int(w/2), 0] >= thres:
#             break
#     result.append(i)
#     for i in range(w):
#         if img[int(h/2), i, 0] >= thres:
#             break
#     result.append(i)
#     for i in reversed(range(w)):
#         if img[int(h/2), i-1, 0] >= thres:
#             break
#     result.append(i)
#     return result
# -----自己的crop和fill的strategy-----
# def my_crop_strategy(img, instance):
#     size = 512
#     overlap = 0.15
#     crop_strategies = []
#     width = instance['imageWidth']
#     height = instance['imageHeight']
#     fill_size = [0, 0, 0, 0]
#     if width < 1000 and height < 10000:
#         num = get_crop_num(height, size, overlap)
#         w = _detect_side_left_edge(img)
#         for i in range(num):
#             crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
#                          w + size] if i != num - 1 else [height - size, height, w, w + size]
#             crop_strategies.append([crop_size, fill_size])
#     elif width > 1000:
#         num = get_crop_num(width, size, overlap)
#         ys = []
#         for obj in instance['shapes']:
#             x, y, w, h = points_to_xywh(obj['points'])
#             ys.append(y + h / 2)
#         ym = np.mean(ys)
#         if ym - size / 2 < 0:
#             ymin, ymax = 0, size
#         elif ym + size / 2 > height:
#             ymin, ymax = height - size, height
#         else:
#             ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
#         for i in range(num):
#             crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
#                          round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size, width]
#             crop_strategies.append([crop_size, fill_size])
#     else:
#         num = get_crop_num(height, size, overlap)
#         w = width / 2 - size / 2
#         for i in range(num):
#             crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
#                          w + size] if i != num - 1 else [height - size, height, w, w + size]
#             crop_strategies.append([crop_size, fill_size])
#     return crop_strategies
#
#
# def _detect_side_left_edge(img):
#     filter = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1] for i in range(100)])
#     h, w = img.shape[0], img.shape[1]
#     for i in range(5, w):
#         pattern = np.array(img[int(h / 2) - 50:int(h / 2) + 50, i - 5:i + 5, 0])
#         if (pattern * filter).sum() / 100 >= 40 and i > 150:
#             break
#     return i - 50
# def my_crop_strategy(img, instance):
#     size = 512
#     overlap = 0.15
#     crop_strategies = []
#     width = round(instance['imageWidth'])
#     height = round(instance['imageHeight'])
#     fill_size = [0, 0, 0, 0]
#     if width < 1000 and height < 10000:
#         num = get_crop_num(height, size, overlap)
#         w = round(_detect_side_left_edge(img))
#         for i in range(num):
#             crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
#                          w + size] if i != num - 1 else [height - size, height, w, w + size]
#             crop_strategies.append([crop_size, fill_size])
#     elif width > 1000:
#         num = get_crop_num(width, size, overlap)
#         ys = []
#         for obj in instance['shapes']:
#             x, y, w, h = points_to_xywh(obj['points'])
#             ys.append(y + h / 2)
#         ym = np.mean(ys)
#         if ym - size / 2 < 0:
#             ymin, ymax = 0, size
#         elif ym + size / 2 > height:
#             ymin, ymax = height - size, height
#         else:
#             ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
#         for i in range(num):
#             crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
#                          round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size, width]
#             crop_strategies.append([crop_size, fill_size])
#     else:
#         num = get_crop_num(height, size, overlap)
#         w = round(width / 2 - size / 2)
#         for i in range(num):
#             crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
#                          w + size] if i != num - 1 else [height - size, height, w, w + size]
#             crop_strategies.append([crop_size, fill_size])
#     return crop_strategies
# def _detect_side_left_edge(img):
#     filter = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1] for i in range(100)])
#     h, w = img.shape[0], img.shape[1]
#     for i in range(5, w):
#         pattern = np.array(img[int(h / 2) - 50:int(h / 2) + 50, i - 5:i + 5, 0])
#         if (pattern * filter).sum() / 100 >= 40 and i > 150:
#             break
#     return i - 50

if __name__ == '__main__':
    x = random.randint(3,3)
    print(x)
    exit(0)
    # path = r'C:\Users\HUAWEI\Desktop\AC'
    # for file in os.listdir(path):
    #     if not file.endswith('.json'): continue
    #     instance = json_to_instance(os.path.join(path, file))
    #     need = False
    #     for obj in instance['shapes']:
    #         label = obj['label']
    #         if label == 'cashang2':
    #             obj['label'] = 'cashang'
    #             need = True
    #             print(label)
    #     if need:
    #         instance_to_json(instance, os.path.join(path, file))
    # exit(0)
    path = r'D:\Data\temp'
    for file in os.listdir(path):
        if not file.endswith('.json'): continue
        instance = json_to_instance(os.path.join(path, file))
        for obj in instance['shapes']:
            print(obj['imageData'])
            obj['imageData'] = None
        instance_to_json(instance, os.path.join(path, file))




































