from dearpygui.core import *
from dearpygui.simple import *
from xml_to_labelme import xml_to_labelme
from labelme_to_yolo import labelme_to_yolo
from coco_to_yolo import split_train_val_test
import os
set_main_window_size(800, 600)

def directory_picker(sender, data):
    select_directory_dialog(callback=apply_selected_directory)

def apply_selected_directory(sender, data):
    log_debug(data)  # so we can see what is inside of data
    directory = data[0]
    folder = data[1]
    set_value("directory", directory)
    set_value("folder", f"{directory}/outputs")
    set_value("labelmepath", f"{directory}/labelme")
  


def xml_labelme(sender, data):

    """
    docstring
    """
    log_debug("translating to label in "+get_value('folder'))
    xml_to_labelme(xml_folder_path=get_value("folder"),img_folder_path=get_value("directory"),item_name='item',json_path=get_value("labelmepath"))

def labelme_yolo(sender, data):
    """
    docstring
    """
    labelme_to_yolo(img_folder_path=get_value("directory"),json_folder_path=get_value("labelmepath"),yolo_folder_path=get_value("directory"))

def split_data(sender, data):
    """
    docstring
    """
    output_dir=os.path.join(get_value("directory"),get_value("target_dir"))

    split_train_val_test(yolo_folder_path=get_value("directory"), target_folder_path=output_dir, test_ratio=get_value("test_ratio"), val_ratio=get_value("val_ratio"))
    log_debug("Data splitted!!!!")

#show_logger()

with window("Main Window"):
    set_value("target_dir","0106")
    set_value("test_ratio",0.15)
    set_value("val_ratio",0.1)
    add_button("Directory Selector", callback=directory_picker)
    add_text("Directory Path: ")
    add_same_line()
    add_label_text("##dir", source="directory", color=[255, 0, 0])
    add_text("outputs Path: ")
    add_same_line()
    add_label_text("##folder", source="folder", color=[255, 0, 0])
    add_text("Labelme Path: ")
    add_same_line()
    add_label_text("##labelmepath", source="labelmepath", color=[255, 0, 0])
    add_text("YOLO Path: ")
    add_same_line()
    add_label_text("##yolopath", source="yolopath", color=[255, 0, 0])
    
    add_button("Translate to labelme", callback=xml_labelme)
    add_button("Translate to yolo", callback=labelme_yolo)

    add_text("Split dataset to Train,Test,Val:")
    add_input_text("target name", source="target_dir")
    add_input_float("Test ratio", source="test_ratio")
    add_input_float("Val ratio", source="val_ratio")
    add_button("Split", callback=split_data)

start_dearpygui(primary_window="Main Window")