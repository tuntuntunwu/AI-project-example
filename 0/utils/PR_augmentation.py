from utils import *
import shutil


def precision_recall_visualize(target_folder_path, cls_id_name_dict, img_boxes_query, 
    saved_folder_name, iou_thres=0.3, hard_thres=0.5, recall=True, precision=True):
    '''
    :param target_folder_path: json文件夹路径
    :param img_boxes_query: 该方法根据json文件名返回box列表
    :return: 可视化漏失、过检结果
    '''
    # 漏失、过检文件夹
    # pr_folder_path = os.path.join(target_folder_path, saved_folder_name)
    pr_folder_path = saved_folder_name
    if not os.path.exists(pr_folder_path): os.makedirs(pr_folder_path)
    img_files = os.listdir(target_folder_path)
    total_objs = 0
    over_detect = 0
    no_detect = 0
    # 遍历img
    for img_file in img_files:
        img_file_path = os.path.join(target_folder_path, img_file)
        # 过滤文件夹和非图片文件
        if not os.path.isfile(img_file_path) or img_file[img_file.rindex('.')+1:] not in IMG_TYPES: continue
        # img_out_path = os.path.join(pr_folder_path, img_file)
        # json_out_path = img_out_path[:img_out_path.rindex('.')] + '.json'
        new_img_file = img_file
        # 读取img的json文件载入instance对象
        try:
            json_file_path = (img_file_path[:img_file_path.rindex('.')] + '.json').replace('/images/', '/jsons/')
            instance = json_to_instance(json_file_path)
        except Exception as e:
            txt_file_path = img_file_path[:img_file_path.rindex('.')] + '.txt'
            txt_file_path = txt_file_path.replace('images', 'labels')
            instance = create_empty_json_instance(img_file_path)
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            width, height = instance['imageWidth'], instance['imageHeight']
            shapes = []
            for line in lines:
                words = line.split(' ')
                cls_id = int(words[0])
                cls_name = cls_id_name_dict[cls_id]
                cx, cy, w, h = float(words[1]) * width, float(words[2]) * height, float(words[3]) * width, float(words[4]) * height
                # confidence = float(words[5])
                # 一个Box代表一个检测目标的xywh、label、confidence
                # boxes.append(Box(cx-w/2, cy-h/2, w, h, cls_name, confidence))
                shape = {}
                shape['label'] = cls_name
                shape['shape_type'] = "rectangle"
                shape['points'] = [[cx-w/2, cy-h/2], [cx+w/2, cy+h/2]]
                shapes.append(shape)

            instance['shapes'] = shapes
            
        predict_boxes = img_boxes_query(img_file_path, cls_id_name_dict)
        # 总待检测目标
        total_objs += len(instance['shapes'])
        # --------------------------全漏失情况--------------------------
        if recall and len(predict_boxes) == 0 and len(instance['shapes']) != 0:
            # for i, obj in enumerate(instance['shapes']):
                # obj['label'] = 'loushi_' + obj['label']
                
            new_img_file = 'loushi_' + img_file
            instance['imagePath'] = new_img_file
            img_out_path = os.path.join(pr_folder_path, new_img_file)
            json_out_path = img_out_path[:img_out_path.rindex('.')] + '.json'
            instance_to_json(instance, json_out_path)
            shutil.copy(img_file_path, img_out_path)
            print('%s has been analyzed.' % (img_file))
            continue
        # --------------------------------------------------------------
        necessary = False
        temp = []
        # 漏失统计
        for obj in instance['shapes']:
            x, y, w, h = points_to_xywh(obj)
            gt_box = Box(x, y, w, h, obj['label'])
            # 漏检 错检
            false_negative, false_label, hard = True, True, True
            for i, predict_box in enumerate(predict_boxes):
                if gt_box.get_iou(predict_box) > iou_thres:
                    false_negative = False
                    temp.append(i)
                    if gt_box.category == predict_box.category:
                        false_label = False
                        if predict_box.confidence > hard_thres:
                            hard = False
                    else:
                        w_category = predict_box.category
            if not recall: continue
            if false_negative:
                # obj['label'] = 'loushi_' + obj['label']
                new_img_file = 'loushi_' + img_file
                instance['imagePath'] = new_img_file
                # 总漏失
                no_detect += 1
            elif false_label:
                # obj['label'] = 'cuowu_%s_' % (w_category) + obj['label']
                new_img_file = 'cuowu_' + img_file
                instance['imagePath'] = new_img_file
            elif hard:
                # obj['label'] = 'hard_' + obj['label']
                new_img_file = 'hard_' + img_file
                instance['imagePath'] = new_img_file
            if false_negative or false_label or hard:
                necessary = True
        # 过检统计
        if precision:
            for i, predict_box in enumerate(predict_boxes):
                if i not in temp:
                    # 总过检
                    over_detect += 1
                    # obj = {'label': 'guojian_'+predict_box.category, 'shape_type': 'rectangle', 'points': [[predict_box.x, predict_box.y], [predict_box.x+predict_box.w, predict_box.y+predict_box.h]]}
                    # instance['shapes'].append(obj)
                    new_img_file = 'guojian_' + img_file
                    instance['imagePath'] = new_img_file
                    necessary = True
        if necessary:
            img_out_path = os.path.join(pr_folder_path, new_img_file)
            json_out_path = img_out_path[:img_out_path.rindex('.')] + '.json'
            instance_to_json(instance, json_out_path)
            shutil.copy(img_file_path, img_out_path)
        print('%s has been analyzed.' % (img_file))
    print('Total objects: %d, missing %d, over-detect: %d' % (total_objs, no_detect, over_detect))




if __name__ == '__main__':
    from PIL import Image
    # 自定义自己的img-boxes对应方法，Box类参考utils.py
    def img_boxes_query(img_file_path, cls_id_name_dict):
        
        txt_folder_path = '/home/xiaozhiheng/temp/detect_result_damian_6/'
        txt_file = img_file_path[img_file_path.rindex(os.sep)+1:img_file_path.rindex('.')] + '.txt'
        txt_file_path = os.path.join(txt_folder_path, txt_file)
        img = Image.open(img_file_path)
        width = img.width
        height = img.height
        boxes = []

        new_name_dict = {v:k for k,v in name_dict.items()}
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return boxes
            # json_file_path = txt_file_path.replace('.txt', '.json')
            # instance = json_to_instance(json_file_path)
            # for shape in instance['shapes']:
            #     # print(shape)
            #     points = shape['points']
            #     label = shape['label']
            #     x, y = points[0][0], points[0][1]
            #     w, h = points[1][0] - points[0][0], points[1][1] - points[0][1]
            #     boxes.append(Box(x, y, w, h, label, 1))
            # return boxes

        # 遍历txt文件中的每一个检测目标
        for line in lines:
            words = line.split(' ')
            cls_id = int(words[0])
            cls_name = cls_id_name_dict[cls_id]
            cx, cy, w, h = float(words[1]) * width, float(words[2]) * height, float(words[3]) * width, float(words[4]) * height
            confidence = float(words[5])
            # 一个Box代表一个检测目标的xywh、label、confidence
            boxes.append(Box(cx-w/2, cy-h/2, w, h, cls_name, confidence))
        return boxes
    name_dict = {0:'bianxing', 1:'chuangkouduoliao', 2:'guoshao', 3:'huashang', 4:'liewen', 5:'maoci', 6:'pengshang', 7:'queliao', 
                 8:'quepeng', 9:'rongjiehen', 10:'shayan', 11:'xuehuazhuang', 12:'zanghua', 13:'zangwu'} 
    # name_dict = {0:'guoshao', 1:'huashang', 2:'liewen', 3:'pengshang', 4:'queliao', 5:'quepeng', 6:'zanghua', 7:'zangwu'}
    precision_recall_visualize(target_folder_path='/home/xiaozhiheng/temp/damian_shuju/data_split_data/images/train/',
                               # 自定义的query方法
                               cls_id_name_dict = name_dict, 
                               img_boxes_query=img_boxes_query,
                               # 保存在图片文件夹下的特定目录名
                               saved_folder_name='/home/xiaozhiheng/temp/pr',
                               # 计算漏失
                               recall=True,
                               # 计算过检
                               precision=True)



























