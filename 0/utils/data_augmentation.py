import json 
# import cv2 
from PIL import Image
import utils
import albumentations as A
import random
import os 
from glob import glob
import numpy as np 
def random_data_aug(img_dir, save_dir):

    save_json_dir = os.path.join(save_dir, 'labelme')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)
    img_path_list = glob(os.path.join(img_dir, '*.jpg'))
    for img_path in img_path_list:
        json_path = img_path.replace('.jpg', '.json')
        img = Image.open(img_path)
        img_name = os.path.basename(img_path) 
        instance = utils.json_to_instance(json_path)
        if 'hard_' in img_path:
            N = 1
            new_img_name = img_name.replace('hard_', 'hard_number') 
        elif 'loushi_' in img_path:
            N = 4
            new_img_name = img_name.replace('loushi_', 'loushi_number') 
        elif 'cuowu_' in img_path:
            N = 2
            new_img_name = img_name.replace('cuowu_', 'cuowu_number')
        elif 'guojian_' in img_path:
            N = 3
            new_img_name = img_name.replace('guojian_', 'guojian_number')
        else:
            N = 1
            new_img_name = img_name
        
        # print(N)
        for i in range(N):
            im = img.copy()
            # image = Image.fromarray(img)
            image = np.asarray(im)
            transform = A.Compose([
                A.MedianBlur(blur_limit=7, always_apply=False, p=0.5),
                A.GaussNoise(var_limit=(5,10), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                ])
            transformed_image = transform(image=image)["image"]
            out_img_name = new_img_name.replace('number', str(i))
            out_json_name = out_img_name.replace('.jpg', '.json')
            instance['imagePath'] = out_img_name
            transformed_image = Image.fromarray(transformed_image)
            transformed_image.save(os.path.join(save_dir, out_img_name))
            utils.instance_to_json(instance, os.path.join(save_json_dir, out_json_name))

if __name__ == '__main__':
    img_dir = '/home/xiaozhiheng/temp/pr_0126'
    save_dir = '/home/xiaozhiheng/temp/aug_pr_0126'

    random_data_aug(img_dir, save_dir)


    


