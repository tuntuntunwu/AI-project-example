from xml.etree import ElementTree as ET
from utils import *
from pathlib import Path
import os
def xml_to_labelme(xml_folder_path: str, img_folder_path: str, item_name: str,json_path:str):
    '''
    :param xml_folder_path: xml文件夹路径
    :param img_folder_path: 图像文件夹绝对路径
    :param item_name: xml文件中目标节点的name
    :return: xml文件转换成labelme json放在图片路径下
    '''


    json_dir = Path(json_path)
    if not json_dir.exists():
        os.mkdir(json_path)
    img_files = os.listdir(img_folder_path)
    # 遍历img
    for img_file in img_files:
        img_file_path = os.path.join(img_folder_path, img_file)
        # 过滤文件夹和非图片文件
        if not os.path.isfile(img_file_path) or img_file[img_file.rindex('.')+1:] not in IMG_TYPES: continue
        # 对应的xml文件
        xml_file_path = os.path.join(xml_folder_path, img_file[:img_file.rindex('.')]+'.xml')
        # 对应的json文件
        print(xml_file_path)
        json_out_path = os.path.join(json_path, img_file[:img_file.rindex('.')]+'.json')
        try:
            root = ET.parse(xml_file_path).getroot()
        except Exception as e:
            # 若为无目标图片
            print('\033[1;33m%s has no xml file in %s. So saved as an empty json.\033[0m' % (img_file, xml_folder_path))
            instance = create_empty_json_instance(img_file_path)
            instance_to_json(instance, json_out_path)
            continue

        # w = int(root.find('size')[0].text)
        img = cv2.imread(img_file_path)
        img_h, img_w, img_c = img.shape
        instance = {'version': '1.0',
                    'shapes': [],
                    'imageData': None,
                    'imageWidth': img_w,
                    'imageHeight': img_h,
                    'imageDepth': img_c,
                    'imagePath': img_file}
        # 一个xml文件中所有的目标
        items = root.iter(item_name)
        for item in items:
            obj = {'label': word_to_pinyin(item[0].text)}
            # 如果有其他标签类别，通过elif添加
            if item.find('bndbox') != None:
                xys = extract_xys(item.find('bndbox'))
                obj['shape_type'] = 'rectangle'
                obj['points'] = [[xys[0], xys[1]], [xys[2], xys[3]]]
            elif item.find('point') != None:
                xys = extract_xys(item.find('point'))
                obj['shape_type'] = 'point'
                obj['points'] = [[xys[0], xys[1]]]
            elif item.find('polygon') != None:
                xys = extract_xys(item.find('polygon'))
                obj['shape_type'] = 'polygon'
                obj['points'] = [[xys[i-1], y] for i,y in enumerate(xys) if i%2==1]
            elif item.find('line') != None:
                xys = extract_xys(item.find('line'))
                # 排除标注小组的重复落点
                points = [[xys[i-1], y] for i,y in enumerate(xys) if i%2==1]
                points_checked = [points[0]]
                for point in points:
                    if point != points_checked[-1]:
                        points_checked.append(point)
                obj['points'] = points_checked
                if len(points_checked) == 1:
                    obj['shape_type'] = 'point'
                elif len(points_checked) == 2:
                    obj['shape_type'] = 'line'
                else:
                    obj['shape_type'] = 'linestrip'
            else:
                print('Please check the xml file to add polygon type!')
                exit(0)
            instance['shapes'].append(obj)
        instance_to_json(instance, json_out_path)
        print('Json created:', instance)

def delete_json(img_folder_path: str):
    '''
    :param img_folder_path: 图像文件夹绝对路径
    :return: 删除该路径下所有json文件
    '''
    for file in os.listdir(img_folder_path):
        if file.endswith('.json'):
            os.remove(os.path.join(img_folder_path, file))

if __name__ == '__main__':
    # 填入xml folder path
    xml_to_labelme(xml_folder_path='./wxq/C件-0429/侧面/outputs',
                   # 填入image folder path
                   img_folder_path='./wxq/C件-0429/侧面',
                   # 填入xml文件中目标标签的name
                   item_name='item',
                   json_path='./wxq')
    # delete_json(img_folder_path='')




















