import os


if __name__ == '__main__':

    src_path = "./文字处理-分词sorted/"
    dst_path = "./sample_statistics/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        os.makedirs(os.path.join(dst_path, "1000+"))
        os.makedirs(os.path.join(dst_path, "500-999"))
        os.makedirs(os.path.join(dst_path, "100-499"))
        os.makedirs(os.path.join(dst_path, "10-99"))
        os.makedirs(os.path.join(dst_path, "2-9"))
        os.makedirs(os.path.join(dst_path, "1"))
    
    for cls in os.listdir(src_path):
        cls_path = os.path.join(src_path, cls)
        sample_number = len(os.listdir(cls_path))
        print(cls)
        print(sample_number)

        if sample_number >= 1000:
            dst_cls_path = os.path.join(dst_path, "1000+", cls)
        elif sample_number >= 500:
            dst_cls_path = os.path.join(dst_path, "500-999", cls)
        elif sample_number >= 100:
            dst_cls_path = os.path.join(dst_path, "100-499", cls)
        elif sample_number >= 10:
            dst_cls_path = os.path.join(dst_path, "10-99", cls)
        elif sample_number >= 2:
            dst_cls_path = os.path.join(dst_path, "2-9", cls)
        else:
            dst_cls_path = os.path.join(dst_path, "1", cls)
        
        if not os.path.exists(dst_cls_path):
            os.makedirs(dst_cls_path)

