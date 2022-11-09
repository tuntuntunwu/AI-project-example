import random
from utils import *


def get_crop_num(img_size, crop_size, overlap):
    '''
    :param img_size: img长或者宽
    :param crop_size: crop的边长
    :param overlap: 相邻框的交并比
    :return: 根据overlap和crop size计算滑框截取个数
    '''
    return math.ceil((img_size-crop_size)/((1-overlap)*crop_size)) + 1

def _random_crop(cx, cy, w, h, size, shift_x_left=0.75, shift_x_right=0.25, shift_y_up=0.75, shift_y_bottom=0.25):
    '''
    :param cx: 目标中心点x
    :param cy: 目标中心点y
    :param w: 图片width
    :param h: 图片height
    :param size: 截图的size
    :param shift_x_left: 截框左边框距离cx的最左随机范围（距离像素/size）
    :param shift_x_right: 截框左边框距离cx的最右随机范围（距离像素/size）
    :param shift_y_up: 截框上边框距离cy的最上随机范围（距离像素/size）
    :param shift_y_bottom: 截框上边框距离cy的最下随机范围（距离像素/size）
    :return: 返回随机截图框
    '''
    # check crop尺寸是否超标
    size_x = w if size > w else size
    size_y = h if size > h else size
    # 截框左边框、上边框距离目标中心点的offset
    ofx, ofy = random.randint(int(size * shift_x_right), int(size * shift_x_left)), random.randint(int(size * shift_y_bottom), int(size * shift_y_up))
    if cy-ofy < 0:
        up, bottom = 0, size_y
    elif cy-ofy+size_y > h:
        up, bottom = h-size_y, h
    else:
        up, bottom = cy-ofy, cy-ofy+size_y
    if cx-ofx < 0:
        left, right = 0, size_x
    elif cx-ofx+size_x > w:
        left, right = w-size_x, w
    else:
        left, right = cx-ofx, cx-ofx+size_x
    return [up, bottom, left, right], [(size-size_y)//2, size-size_y-(size-size_y)//2, (size-size_x)//2, size-size_x-(size-size_x)//2]

# 根据过检和漏失的增强截图策略
def aug_crop_strategy(img, instance):
    # 截图尺寸
    size = 768
    # 过检增强倍数
    precision_aug = 2
    # 漏失增强倍数
    recall_aug = 4
    # 错误、难样本增强倍数
    hard_aug = 2
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        label = obj['label']
        cx, cy = points_to_center(obj)
        cx, cy = int(cx), int(cy)
        # if label.startswith('guojian'):
        #     for i in range(precision_aug):
        #         crop_strategies.append(_random_crop(cx, cy, w, h, size))
        # elif label.startswith('loushi'):
        #     for i in range(recall_aug):
        #         crop_strategies.append(_random_crop(cx, cy, w, h, size))
        # elif label.startswith('hard') or label.startswith('cuowu'):
        #     for i in range(hard_aug):
        #         crop_strategies.append(_random_crop(cx, cy, w, h, size))
        crop_strategies.append(_random_crop(cx, cy, w, h, size))
    return crop_strategies

def val_crop_strategy(img, instance):
    size = 512
    overlap = 0.10
    d = 100
    crop_strategies = []
    width = round(instance['imageWidth'])
    height = round(instance['imageHeight'])
    fill_size = [0, 0, 0, 0]
    if width < 1000 and height < 10000:
        num = get_crop_num(height - 2 * d, size, overlap)
        w = round(_detect_side_left_edge(img))
        for i in range(num - 1):
            crop_size = [round(i * (1 - overlap) * size) + d, round(i * (1 - overlap) * size) + size + d, w,
                         w + size]
            crop_strategies.append([crop_size, fill_size])
    elif width > 1000:
        num = get_crop_num(width, size, overlap)
        ys = []
        for obj in instance['shapes']:
            x, y, w, h = points_to_xywh(obj)
            ys.append(y + h / 2)
        ym = np.mean(ys)
        if ym - size / 2 < 0:
            ymin, ymax = 0, size
        elif ym + size / 2 > height:
            ymin, ymax = height - size, height
        else:
            ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
        for i in range(num):
            crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
                         round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size,
                                                                                       width]
            crop_strategies.append([crop_size, fill_size])
    else:
        num = get_crop_num(height - 2 * d, size, overlap)
        w = round(width / 2 - size / 2 - 65)
        for i in range(num - 1):
            crop_size = [round(i * (1 - overlap) * size) + d, round(i * (1 - overlap) * size) + size + d, w,
                         w + size]
            crop_strategies.append([crop_size, fill_size])
    return crop_strategies

def my_crop_strategy(img, instance):
    size = 512
    overlap = 0.15
    crop_strategies = []
    width = round(instance['imageWidth'])
    height = round(instance['imageHeight'])
    fill_size = [0, 0, 0, 0]
    if width < 1000 and height < 10000:
        num = get_crop_num(height, size, overlap)
        w = round(_detect_side_left_edge(img))
        for i in range(num):
            crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
                         w + size] if i != num - 1 else [height - size, height, w, w + size]
            crop_strategies.append([crop_size, fill_size])
    elif width > 1000:
        num = get_crop_num(width, size, overlap)
        ys = []
        for obj in instance['shapes']:
            x, y, w, h = points_to_xywh(obj)
            ys.append(y + h / 2)
        ym = np.mean(ys)
        if ym - size / 2 < 0:
            ymin, ymax = 0, size
        elif ym + size / 2 > height:
            ymin, ymax = height - size, height
        else:
            ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
        for i in range(num):
            crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
                         round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size,
                                                                                       width]
            crop_strategies.append([crop_size, fill_size])
    else:
        num = get_crop_num(height, size, overlap)
        w = round(width / 2 - size / 2)
        for i in range(num):
            crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
                         w + size] if i != num - 1 else [height - size, height, w, w + size]
            crop_strategies.append([crop_size, fill_size])
    return crop_strategies

def _detect_side_left_edge(img):
    filter = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1] for i in range(100)])
    h, w = img.shape[0], img.shape[1]
    for i in range(5, w):
        pattern = np.array(img[int(h / 2) - 50:int(h / 2) + 50, i - 5:i + 5, 0])
        if (pattern * filter).sum() / 100 >= 40 and i > 150:
            break
    return i - 50

def huawei_ac_crop_strategy(img, instance):
    size = 256
    aug = 4
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        cx, cy = points_to_center(obj)
        cx, cy = int(cx), int(cy)
        for i in range(aug):
            crop_strategies.append(_random_crop(cx, cy, w, h, size))
    return crop_strategies

# 检测特定标签的截图策略
def check_crop_strategy(img, instance):
    # 截图尺寸
    size = 512
    # 需要查看的labels
    check_list = ['shuiyin', 'youmo']
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        label = obj['label']
        if label not in check_list: continue
        cx, cy = points_to_center(obj)
        cx, cy = int(cx), int(cy)
        crop_strategies.append(_random_crop(cx, cy, w, h, size, 0.5, 0.5, 0.5, 0.5))
    return crop_strategies

# 聚类截图策略
def clustering_crop_strategy(img, instance):
    # 截图尺寸
    size = 512
    crop_strategies = []
    # 用来存放截取过的obj
    added = []
    objs = instance['shapes']
    num = len(objs)
    for i, obj in enumerate(objs):
        # 如果obj被截取过，continue
        if obj in added: continue
        # 当前聚类的外边框
        current_box = Box(*points_to_xywh(obj))
        # 开始搜寻聚类的objs
        for j in range(i+1, num):
            # 下一个obj
            next_obj = objs[j]
            # 如果下一个obj被截取过，continue
            if next_obj in added: continue
            next_box = Box(*points_to_xywh(next_obj))
            # 将下一个obj融合进当前的聚类的外边框
            combine_box = _combine_boxes(current_box, next_box)
            # 如果下一个obj不适合聚类，continue
            if combine_box.w > size or combine_box.h > size: continue
            # 聚类完成，更新当前的聚类的外边框
            current_box = combine_box
            # 将下一个obj放入added列表
            added.append(next_obj)

    return crop_strategies

def _combine_boxes(box1, box2):
    pass


































