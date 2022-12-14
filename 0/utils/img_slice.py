import copy
from concurrent.futures.thread import ThreadPoolExecutor
from img_slice_utils import *


def imgs_crop_and_fill(img_folder_path, crop_stategy, output_path='outputs', empty_check=True, num_worker=8, iou_thres=0.2):
    '''
    :param img_folder_path: 图片文件夹绝对路径
    :param crop_stategy: 截取策略
    :param empty_check: 是否检查空截框
    :param num_worker: 线程数
    :param output_path: 保存路径
    :param iou_thres: 截图时iou阈值
    :return: 整个文件夹下的图片截图和填充
    '''
    thread_pool = ThreadPoolExecutor(max_workers=num_worker)
    print('Thread Pool is created!')
    # 检查输出路径文件夹是否存在
    if not os.path.exists(output_path): os.makedirs(output_path)
    for img_name in os.listdir(img_folder_path):
        img_file_path = os.path.join(img_folder_path, img_name)
        # 过滤文件夹和非图片文件
        # if not os.path.isfile(img_file_path) or img_name[img_name.rindex('.')+1:] not in IMG_TYPES: continue
        if not os.path.isfile(img_file_path) or img_name[img_name.rindex('.')+1:] not in IMG_TYPES:continue
        thread_pool.submit(img_crop_and_fill, img_file_path, crop_stategy, output_path, empty_check, iou_thres)
    thread_pool.shutdown(wait=True)

# 单张图片的截图和填充，用来作为多线程的输入方法
def img_crop_and_fill(img_file_path, crop_strategy, output_path, empty_check, iou_thres):
    img = cv2.imread(img_file_path)
    try:
        instance = json_to_instance((img_file_path[:img_file_path.rindex('.')] + '.json').replace('images', 'jsons'))
    except FileNotFoundError:
        # 有些图片没有对应的json文件，或表示该图片无目标
        print('\033[1;33m%s has no json file...\033[0m' % (img_file_path))
        instance = create_empty_json_instance(img_file_path)
    instance_clean(instance)
    instance_points_to_polygon(instance)
    crops = crop_strategy(img, instance)
    for crop in crops:
        crop_size, fill_size = crop[0], crop[1]
        try:
            init_crop_and_fill(img, instance, crop_size, fill_size, output_path, empty_check, iou_thres)
        except Exception as e:
            # 有些图片crop报错需要log
            print('\033[1;31m%s fails in cropping %s, due to %s.\033[0m' % (img_file_path, crop_size, e.with_traceback()))

# 单张图片的截图和填充，要求输入已经读取的img和json instance
def init_crop_and_fill(img, instance, crop_size, fill_size, output_path, empty_check, iou_thres):
    if empty_check and crop_is_empty(instance, crop_size, iou_thres): return
    # 新图片名、json文件名，新图片路径，json文件路径
    offset_x, offset_y = crop_size[2]-fill_size[2], crop_size[0]-fill_size[0]
    img_new_name = instance['imagePath'].replace('.', '_%d_%d.' % (offset_x, offset_y))
    # print(img_new_name)
    # img_new_name = merge_img_name(img_new_name)
    json_new_name = img_new_name[:img_new_name.rindex('.')] + '.json'
    img_new_path = os.path.join(output_path, img_new_name)
    json_new_path = os.path.join(output_path, json_new_name)
    instance_new = {'version': '1.0', 'imageData': None,
                    'imageWidth': crop_size[3] - crop_size[2] + fill_size[2] + fill_size[3],
                    'imageHeight': crop_size[1] - crop_size[0] + fill_size[0] + fill_size[1],
                    'imageDepth': img.shape[2],
                    'imagePath': img_new_name, 'shapes': copy.deepcopy(instance['shapes'])}
    # 先截图后填充
    img_crop = img[crop_size[0]: crop_size[1], crop_size[2]: crop_size[3]]
    update_objs_in_crop(instance_new, crop_size, iou_thres)
    img_new = cv2.copyMakeBorder(img_crop, fill_size[0], fill_size[1], fill_size[2], fill_size[3], cv2.BORDER_REPLICATE)
    cv2.imwrite(img_new_path, img_new)
    for obj in instance_new['shapes']:
        for point in obj['points']:
            point[0] -= offset_x
            point[1] -= offset_y
    instance_to_json(instance_new, json_new_path)
    print(img_new_name, ' is done!')

# 合并name的后缀信息
def merge_img_name(img_name: str):

    
    info = img_name[img_name.index('_')+1:img_name.rindex('.')].split('_')
    if len(info) == 2: return img_name
    offset_x = int(info[0]) + int(info[2])
    offset_y = int(info[1]) + int(info[3])
    return img_name[:img_name.index('_')] + '_%d_%d'%(offset_x, offset_y) + img_name[img_name.rindex('.'):]

# instance中的shapes字段为原instance中shapes字段的深拷贝
# 此方法更新instance中的shapes字段
def update_objs_in_crop(instance, crop_size, iou_thres=0.2):
    shapes_new = []
    shapes = instance['shapes']
    # 遍历shapes中的目标objs
    for obj in shapes:
        points = obj['points']
        shape_type = obj['shape_type']
        # 目标不在crop区域，continue
        if not obj_in_crop(obj, crop_size, iou_thres):
            continue
        # 目标在crop区域，开始更新坐标
        points_new = []
        # 四条截边
        bounds = [[crop_size[2], crop_size[0], crop_size[3], crop_size[0]],  # (xmin, ymin, xmax, ymin)
                  [crop_size[3], crop_size[0], crop_size[3], crop_size[1]],  # (xmax, ymin, xmax, ymax)
                  [crop_size[3], crop_size[1], crop_size[2], crop_size[1]],  # (xmax, ymax, xmin, ymax)
                  [crop_size[2], crop_size[1], crop_size[2], crop_size[0]]]  # (xmin, ymax, xmin, ymin)
        # four_points = [[crop_size[2], crop_size[0]], [crop_size[3], crop_size[0]], [crop_size[3], crop_size[1]], [crop_size[2], crop_size[1]]]
        # xywh = points_to_xywh(obj)
        for i, point in enumerate(points):
            if point_in_crop(point, crop_size):
                if (i != 0 or shape_type == 'polygon') and (not point_in_crop(points[i-1], crop_size)):
                    for bound in bounds:
                        cross_point = get_cross_point(point[0], point[1], points[i-1][0], points[i-1][1], *bound)
                        if cross_point != None:
                            points_new.append(cross_point)
                            break
                points_new.append(point)
            elif (i != 0 or shape_type == 'polygon') and point_in_crop(points[i-1], crop_size):
                for bound in bounds:
                    cross_point = get_cross_point(point[0], point[1], points[i-1][0], points[i-1][1], *bound)
                    if cross_point != None:
                        points_new.append(cross_point)
                        break
            elif (i != 0 or shape_type == 'polygon') and (not point_in_crop(points[i-1], crop_size)):
                temp = []
                for bound in bounds:
                    cross_point = get_cross_point(point[0], point[1], points[i-1][0], points[i-1][1], *bound)
                    if cross_point != None:
                        temp.append(cross_point)
                if len(temp) == 0: continue
                if (temp[0][0]-point[0])**2+(temp[0][1]-point[1])**2 > (temp[1][0]-point[0])**2+(temp[1][1]-point[1])**2:
                    points_new.append(temp[0])
                    points_new.append(temp[1])
                else:
                    points_new.append(temp[1])
                    points_new.append(temp[0])
        obj['points'] = points_new
        shapes_new.append(obj)
    instance['shapes'] = shapes_new

if __name__ == '__main__':
    # 在这里定义自己的crop和fill的strategy
    # crop_strategy方法：根据img对象和json instance，给出crop和fill的范围，遵循上下左右
    # return [[crop1, fill1], [crop2, fill2],...]
    # crop: [top, bottom, left, right]   fill: [top, bottom, left, right]
    # img为opencv读取的图片对象，instance为json对象
    # 内置的crop_strategy方法在img_slice_utils.py中
    def define_my_crop_strategy(img, instance):
        pass
    # 截图并保存
    imgs_crop_and_fill(img_folder_path='/home/xiaozhiheng/temp/new_data/20210323/labeled_image/damian',
                       # 自定义的截图策略
                       crop_stategy=aug_crop_strategy,
                       # 截图输出路径
                       output_path='/home/xiaozhiheng/temp/new_data/20210323/labeled_image/crop_damian',
                       # 自动滤去不含检测目标的截图框
                       empty_check=True,
                       # 被截断的检测目标的面积比阈值，低于阈值将不计入截图框中
                       iou_thres=0.3)
































