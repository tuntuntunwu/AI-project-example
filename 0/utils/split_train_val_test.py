from coco_to_yolo import split_train_val_test

if __name__ == '__main__':
    # 按比例分训练、验证、测试集
    split_train_val_test(img_folder_path=r'../heiheihei/C:\Users\HUAWEI\Desktop\AC\crop',
                         target_folder_path=r'C:\Users\HUAWEI\Desktop\AC\ac',
                         test_ratio=0.1,
                         val_ratio=1)