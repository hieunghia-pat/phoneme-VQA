import os
import numpy as np
import pandas as pd


def textlayout_ocr_adapt(ocr_root, h_scale=1000, w_scale=1000):
    image_id = []
    info = []

    for ocr_file in os.listdir(ocr_root):
        info_ = {}
        image_id.append(float(ocr_file[:-4]))
        info_['image_id'] = float(ocr_file[:-4])

        o = np.load(os.path.join(ocr_root, ocr_file), allow_pickle=True).tolist()

        info_['texts'] = o['texts']


        pre_bboxes = o['boxes'].tolist()

        bboxes = []

        height = 1#o['height']
        width = 1#o['width']

        for i in range(len(pre_bboxes)):

            top_left_x = float(pre_bboxes[i][0]*1.0/width*w_scale)
            top_left_y = float(pre_bboxes[i][1]*1.0/height*h_scale)
            bottom_right_x = float(pre_bboxes[i][2]*1.0/width*w_scale)
            bottom_right_y = float(pre_bboxes[i][3]*1.0/height*h_scale)

            bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        info_['bboxes'] = bboxes


        info.append(info_)


    ocr_df = pd.DataFrame({'image_id':image_id, 'obj_info':info})
    ocr_df['texts'] = ocr_df['obj_info'].apply(lambda x: list(x['texts']))
    ocr_df['bboxes'] = ocr_df['obj_info'].apply(lambda x: list(x['bboxes']))
    return ocr_df



def textlayout_obj_adapt(obj_root, h_scale=1000, w_scale=1000):
    image_id = []
    info = []

    for obj_file in os.listdir(obj_root):
        info_ = {}
        image_id.append(float(obj_file[:-4]))
        info_['image_id'] = float(obj_file[:-4])

        o = np.load(os.path.join(obj_root, obj_file), allow_pickle=True).tolist()

        info_['object_list'] = o['object_list']


        pre_bboxes = o['region_boxes'].tolist()

        bboxes = []

        height = o['height']
        width = o['width']

        for i in range(len(pre_bboxes)):

            top_left_x = float(pre_bboxes[i][0]*1.0/width*w_scale)
            top_left_y = float(pre_bboxes[i][1]*1.0/height*h_scale)
            bottom_right_x = float(pre_bboxes[i][2]*1.0/width*w_scale)
            bottom_right_y = float(pre_bboxes[i][3]*1.0/height*h_scale)

            bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        info_['bboxes'] = bboxes


        info.append(info_)


    obj_df = pd.DataFrame({'image_id':image_id, 'obj_info':info})
    obj_df['obj_labels'] = obj_df['obj_info'].apply(lambda x: list(x['object_list']))
    obj_df['obj_bboxes'] = obj_df['obj_info'].apply(lambda x: list(x['bboxes']))
    return obj_df