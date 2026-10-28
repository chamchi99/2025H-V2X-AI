import argparse
import datetime
import pandas as pd
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math 
import time
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo_comb_v3 import create_tracker
from trackers.strong_sort.reid_multibackend import ReIDDetectMultiBackend

from colordetect import ColorDetect
from math import sqrt

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

from PIL import Image
######
from speed import *
import geopy.distance as geopydist
from ch_setting_comb import *
######

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'best_20221215.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x1_0_vehicle.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.8,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    
    source = str(source)

    # print(1)
    # print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    names[0]='Bus'
    names[1]='Car'
    names[2]='Truck'

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # print(2)
        # print(source)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        for frame_idx, v in enumerate(dataset):
            print(frame_idx)
        # dataset_z = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        # dataset_1 = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        nr_sources = 3
    else:

        if 'cctv02' in source:
            source_z = source.replace('cctv02', 'cctv03')
            source_1 = source.replace('cctv02', 'cctv01')
        elif 'ch02' in source:
            source_z = source.replace('ch02', 'ch03')
            source_1 = source.replace('ch02', 'ch01')


        
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        dataset_z = LoadImages(source_z, img_size=imgsz, stride=stride, auto=False)
        dataset_1 = LoadImages(source_1, img_size=imgsz, stride=stride, auto=False)
        nr_sources = 3

    # print('img size **********')
    # print(dataset.img_size, dataset_z.img_size, pt)
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

#############################################################################
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    #     nr_sources = 1
    # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
#############################################################################


    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):

        if i == 0:
            tracker = create_tracker('ocsort', reid_weights, device, half)
        if i == 1:
            tracker = create_tracker('strongsort_rotate', reid_weights, device, half)
        if i == 2:
            tracker = create_tracker('ocsort', reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()

# #############################################################################
#         tracker = create_tracker(tracking_method, reid_weights, device, half)
#         tracker_list.append(tracker, )
#         if hasattr(tracker_list[i], 'model'):
#             if hasattr(tracker_list[i].model, 'warmup'):
#                 tracker_list[i].model.warmup()

#############################################################################

    outputs = [None] * nr_sources

    #############################
    id_list = []
    bot_list = []
    frame_list = []

    lane_1 = [] 
    lane_2 = []
    lane_3 = []

    mask = []
    global_point = []

    speed_lane_1 = []
    speed_lane_2 = []
    speed_lane_3 = []

    traffic_lane_1 = []
    traffic_lane_2 = []
    traffic_lane_3 = []
    avg_speed_lane_1 = []
    avg_speed_lane_2 = []
    avg_speed_lane_3 = []
    avg_speed_lane_1 = -999
    avg_speed_lane_2 = -999
    avg_speed_lane_3 = -999
    avg_speed_lane = [-999,-999,-999]
    avg_speed_list = [-999,-999,-999]

    frame_index = 0

    ch01_exist = False
    ch02_exist = False
  
    # json
    cctv_traffic_json = "cctv-traffic.json"
    cctv_traffic_vds_json = "cctv-traffic-vds.json"
    entry_json = "entry.json"
    cctv_traffic = {}
    cctv_traffic = []
    cctv_traffic_vds = {}
    cctv_traffic_vds = []
    entry = []

    frame_index = 0
    frame_width = 1080
    frame_height = 1920

    frame = np.zeros((frame_height, frame_width, 3), dtype=np.int8)
    
    draw_inference = True
    draw_coordinate = True
    up_line, down_line, vup_line, vdown_line, mask, lane_1, lane_2, lane_3 = ch_line_rotate(0, 0)

    ###########################
    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup


    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources


    H = np.load('homography_matrix_new_2.npy')
    id_transition_dic = {}

    av_car_flag = False
    av_car_flag2 = False
    AV_flag = False
    eta_flag = True
    AV_eta = 0 
    av_car_id = -1    
    av_classifier_weight = WEIGHTS / 'osnet_x1_0_vehicle.engine'
    av_classifier = ReIDDetectMultiBackend(av_classifier_weight, device=device)

    query_img = [np.asarray(Image.open('query_image/av_query.jpg'))]
    # query_img = [np.asarray(Image.open('query_image/bmw_5.jpg'))]
    query_feature = av_classifier(query_img).detach().cpu().numpy()
    query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)


    for frame_idx, (full, zoom,ch1) in enumerate(zip(dataset, dataset_z,dataset_1)):

            
        cctv_traffic.clear()
        entry.clear()
        cctv_traffic_vds.clear()
    # for frame_idx, (full, zoom,ch1) in enumerate(dataset):
        print('frame idx')
        print(frame_idx)
        # if frame_idx % 2 ==0:
        #     # print('passed')
        #     continue
        merge_count = 0
        f_start = time.time()
        path, im, im0s, vid_cap_f, s_f = full
        path_z, im_z, im0s_z, vid_cap_z, s_z = zoom
        path_1, im_1, im0s_1, vid_cap_1, s_1 = ch1


############################################3
    # for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
###############################3

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        im_z = torch.from_numpy(im_z).to(device)
        im_z = im_z.half() if half else im_z.float()  # uint8 to fp16/32
        im_z /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im_z.shape) == 3:
            im_z = im_z[None]  # expand for batch dim

        im_1 = torch.from_numpy(im_1).to(device)
        im_1 = im_1.half() if half else im_1.float()  # uint8 to fp16/32
        im_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im_1.shape) == 3:
            im_1 = im_1[None]  # expand for batch dim

###############################################
        # if len(im.shape) == 3:
        #     im = im[None]  # expand for batch dim
################################################


        t2 = time_sync()
        dt[0] += t2 - t1
    
        
        t_im = torch.cat((im,im_z,im_1), dim=0)
        t_pred = model(t_im, augment=augment, visualize=False)

        pred = t_pred[0].unsqueeze(0)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        pred_z = t_pred[1].unsqueeze(0)
        pred_z = non_max_suppression(pred_z, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        pred_1 = t_pred[2].unsqueeze(0)
        pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        
        '''
        pred = model(im, augment=augment, visualize=increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        pred_z = model(im_z, augment=augment,visualize=increment_path(save_dir / Path(path_z[0]).stem, mkdir=True) if visualize else False)
        pred_z = non_max_suppression(pred_z, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        pred_1 = model(im_1, augment=augment,visualize=increment_path(save_dir / Path(path_1[0]).stem, mkdir=True) if visualize else False)
        pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        '''


# '''

#         # Inference
#         visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
#         pred = model(im, augment=augment, visualize=visualize)
#         t3 = time_sync()
#         dt[1] += t3 - t2

#         # Apply NMS
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#         #pred boxes

# '''
        t3 = time_sync()
        dt[1] += t3 - t2

        # if frame_idx == 1:
        #     print(pred)


        #     for pred_idx in range(len(pred)):
        #         print(pred_idx)
        #         print(pred[pred_idx].shape)
        #     img = Image.fromarray(im)
        
        
        dt[2] += time_sync() - t3


        # total_pred = [pred_z, pred]
        # # total_pred = [pred, pred_z]
        # total_path = [path_z, path]
        # total_im = [im_z, im]
        # total_im0s = [im0s_z, im0s]
        # total_vid_cap = [vid_cap_z, vid_cap_f]
        # total_s = [s_z, s_f]

        # save_pathes = [0] * nr_sources
        # matching_can = []

        total_pred = [pred_z, pred, pred_1]
        # total_pred = [pred, pred_z]
        total_path = [path_z, path, path_1]
        total_im = [im_z, im, im_1]
        total_im0s = [im0s_z, im0s, im0s_1]
        total_vid_cap = [vid_cap_z, vid_cap_f,vid_cap_1]
        total_s = [s_z, s_f, s_1]


        save_pathes = [0] * nr_sources
        matching_can = []

        # print('')
        # print(pred)
        # print('') 
        # Process detections

        reid_time = 0
        track_time = 0


        for i, det in enumerate(total_pred):  # detections per image
            det = det[0]
# '''
#         for i, det in enumerate(pred):  # detections per image
#         '''
            seen += 1
            if webcam:  # nr_sources >= 1
                # p, im0, _ = path[i], im0s[i].copy(), dataset.count
                # p = Path(p)  # to Path
                # s += f'{i}: '
                # txt_file_name = p.name
                # save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...

                p, im0, _ = total_path[i], total_im0s[i].copy(), getattr(dataset, 'frame', 0)
                # print(p)
                # print(len(im0))
                # im0 = im0[0]
                # print('path1')
                # print(p)
                p = Path(p)  # to Path
                # print('path2')
                # print(p)
                # print(p.stem)
                # print(str(save_dir))
                # print(p.name)
                # print(p.stem+str(i)+'.mp4')
                vid_name = p.stem+str(i)+'.mp4'
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / vid_name)  # im.jpg, vid.mp4, ...
                # print(txt_file_name)
                # print(save_path)

            else:

                p, im0, _ = total_path[i], total_im0s[i].copy(), getattr(dataset, 'frame', 0)

# '''
#                 p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
#                 '''
                p = Path(p)  # to Path

                # video file
                if source.endswith(VID_FORMATS):

                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            # print('im0 shape')
            # print(im0.shape)

            # im0 shape
            # (1080, 1920, 3)
            curr_frames[i] = im0
            save_pathes[i] = save_path

            s = total_s[i]
            vid_cap = total_vid_cap[i]
# '''
#             curr_frames[i] = im0
# '''
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            # annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            # cv2.imwrite("rtsp_img.jpg",im0)
            # print(im0)
            annotator = Annotator(im0, line_width=3, pil=not ascii,font_size = 3)

# '''
#             # if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
#             #     if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
#             #         tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
# '''


            if det is not None and len(det):
                # 
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy
                det[:, :4] = scale_coords(total_im[i].shape[2:], det[:, :4], im0.shape).round()  # xyxy


                #####################################################3
                #####################################################3
                #####################################################3
                #####################################################3
                #####################################################3
                # print('det')
                # print(det)
                # print('img')
                # print(im0.shape)
                # if im0.shape != (1080, 1920, 3):
                #     print('check img size')
                #     continue


                #####################################################3
                #####################################################3
                #####################################################3



                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                if i ==0:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    # print('ouput 0')
                    # print(outputs[i])
                    zoom_av_candidate = []


                elif i ==1:
                    outputs[i],lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 ,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane  = tracker_list[i].update(det.cpu(), im0,lane_1, lane_2, lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane,mask, frame_idx )
                    # print('ouput 1')
                    # print(outputs[i])

                elif i ==2:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    # print('ouput 2')
                    # print(outputs[i])



                # outputs[i], lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3  = tracker_list[i].update(det.cpu(), im0, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_list,frame_idx )
                t5 = time_sync()
                dt[3] += t5 - t4

                track_time += t5 - t4


                if i == 0 :
                    zoom_img_crops = []
                    zoom_av_candidate = []
                    zoom_color = []

                    for output in outputs[i]:
                        bboxes = output[0:4]
                        zoom_id = output[4] * 100
                        zoom_img_crops.append(im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])])
                        zoom_av_candidate.append(zoom_id)


#############################################3color classifier####################33
                        # for crop_idx,zoom_img_crop in enumerate(zoom_img_crops):
                        #     # print('crop type')
                        #     # print(type(zoom_img_crop))
                        #     cv2.imwrite("zoom_crop.jpg",zoom_img_crop)
                        #     my_image = ColorDetect("zoom_crop.jpg")
                        #     extracted_color = list(my_image.get_color_count(color_format="rgb"))
                        #     second_color = extracted_color[-2][1:-1]
                        #     second_color = [int(float(x)) for x in second_color.split(", ")]
                        #     first_color = extracted_color[-1][1:-1]
                        #     first_color = [int(float(x)) for x in first_color.split(", ")]
                        #     zoom_color.append(nearest_colour( colours, first_color )[3] )
                        # ############################################  
                            
                        #     print('crop color')
                        #     print('crop color')
                        #     print('crop color')
                        #     print('crop color')
                        #     print('crop color')
                        #     print('crop color')

                        #     print(zoom_av_candidate[crop_idx])
                        #     print( nearest_colour( colours, second_color ) ) # white
                        #     print( nearest_colour( colours, first_color ) ) # white
                        #     print( nearest_colour( colours, first_color )[3] ) # white
                        #     print(zoom_color)
#################################################################################################################           




                if i == 0 and (not av_car_flag):
                    zoom_img_crops = []
                    zoom_av_candidate = []
                    for output in outputs[i]:

                        bboxes = output[0:4]
                        zoom_id = output[4] * 100
                        bbox_cx = (bboxes[2] + bboxes[0])/2
                        bbox_cy = (bboxes[3] + bboxes[1])/2
                        if 200 < bbox_cy and bbox_cy < 800:
                            if (200 < bbox_cx) and (bbox_cx < 1400):
                                zoom_img_crops.append(im0[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])])
                                zoom_av_candidate.append(zoom_id)


                        if len(zoom_av_candidate) > 0:
                            z_features = av_classifier(zoom_img_crops).detach().cpu().numpy()
                            z_features = z_features / np.linalg.norm(z_features, axis=1, keepdims=True)
                            dist = 1. - np.dot(query_feature, z_features.T)
                            # print('distance')
                            # print(dist)

                            for dist_idx in range(np.shape(dist)[1]):
                                if dist[0, dist_idx] < 0.40:


                                    av_car_flag = True
                                    av_car_id = zoom_av_candidate[dist_idx]
                                    zoom_av_candidate.append(88888)
                                    print(av_car_id)

                if i == 0:

                    merge_count = len(outputs[i])

                    for output in outputs[i]:
                        bboxes = output[0:4]
                        zoom_id = output[4] * 100
                        av_car_flag2 = False
                        if output[4] == 88888:
                            zoom_id = output[4]
                            av_car_flag2 = True

                        bbox_cx = (bboxes[2] + bboxes[0])/2
                        bbox_cy = (bboxes[3] + bboxes[1])/2
                        if zoom_id not in id_transition_dic.values():
                            if 800 < bbox_cy:
                                if (200 < bbox_cx) and (bbox_cx < 1400):
                                    matching_can.append((bbox_cx, bbox_cy, zoom_id))

    
                if i == 1:
                    row_ind, col_ind = None, None
                    if len(matching_can) > 0:
                        reid_start = time.time()
                        before = []
                        zoom_can_id = []
                        for can in matching_can:
                            before.append([can[0], can[1], 1])
                            zoom_can_id.append(can[2])

                        before = np.array(before).transpose()
                        after = np.matmul(H, before)
                        after = after / after[2, :]
                        after = after[:2, :]
                        after = np.round(after, 0)
                        after = after.transpose()

                        normal_can_id = []
                        output_re = []
                        for output in outputs[i]:
                            bboxes = output[0:4]
                            normal_id = output[4]
                            center_x = (bboxes[2] + bboxes[0])/ 2
                            center_y = (bboxes[3] + bboxes[1])/ 2

                            bot_x = output[11]
                            bot_y = output[12]

                            if normal_id not in id_transition_dic.keys():
                                if (80 < center_y) and (center_y < 268):
                                    if (237 < center_x) and (center_x < 700):
                                        output_re.append([center_x, center_y])
                                        normal_can_id.append(normal_id)
    
                        output_re = np.array(output_re)

                        if len(normal_can_id) > 0:
                            cost = distance_matrix(after, output_re)
                            row_ind, col_ind = linear_sum_assignment(cost)
                            for idx, v in enumerate(row_ind):
                                id_transition_dic[normal_can_id[idx]] = zoom_can_id[v]

                        reid_end = time.time()
                        reid_time = reid_end - reid_start

                # if len(outputs)==0:
                #     # print(outputs)
                #     continue


                
                
                # if not "traffic" in cctv_traffic:
                #     cctv_traffic['traffic'] = []

                if outputs[1] is not None:
                    if len(outputs[1]) == 0  :
                        ch02_exist = False
                    elif len(outputs[1]) > 0 :
                        ch02_exist = True


                if outputs[2] is not None:

                    if len(outputs[2]) == 0 :
                        ch01_exist = False
                    elif len(outputs[2]) > 0 :
                        ch01_exist = True


                if len(outputs[i]) > 0:

                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                        AV_flag = False
# '''
#                     for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
#                         '''
                        # print(output)
                        # print(conf)
                        bboxes = output[0:4]
                        # bbox = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if i ==1:
                            final_speed = output[7]
                            
                            bbox_x = output[8]
                            bbox_y = output[9]


                        if i == 0 and id != 88888:
                            id = id * 100
                            if av_car_flag and id == av_car_id:
                                id = 88888


                        if i == 1 and id in id_transition_dic.keys():
                            id = id_transition_dic[id]
                            
                            if av_car_flag and id == av_car_id:
                                id = 88888
                                AV_flag = True
############################3
############################3
############################3
############################3
############################3
                        if len(output)>8:
                            
                            # print(output[7])

                            # cctv_traffic.append({"traffic":{"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS(output[8],output[9])[0], toWGS(output[8],output[9])[1]],"speed":output[7],"lane":output[10],"timestamp":int(time.time())}})
                            
                            ##################
                            #####################
                            cctv_traffic.append({"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS(output[8],output[9])[0], toWGS(output[8],output[9])[1]],"speed":output[7],"lane":output[10],"timestamp":int(time.time())})
                            ##################
                            #############3
                            entry.append({"cctv":i+1,"class":int(output[5]),"id":id,"est_lane":output[10],"eta":eta(toWGS(output[8],output[9]),output[7]),"timestamp":int(time.time())})
                        #  cctv_traffic.append({"cctv":i+1,"class":int(output[5]),"id":output[4],"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS(output[8],output[9])[0], toWGS(output[8],output[9])[1]],"speed":output[7],"lane":output[10],"eta":eta(toWGS(output[8],output[9]),output[7]),"timestamp":int(time.time())})
############################3
############################3
############################3
############################3

                        if i == 0:
                            print(i)
                            # cctv_traffic.append({"traffic":{"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS_ch1(output[0],output[1])[0], toWGS_ch1(output[0],output[1])[1]],"speed": None ,"lane": None ,"timestamp":int(time.time())}})
                            cctv_traffic.append({"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS_ch1(output[0],output[1])[0], toWGS_ch1(output[0],output[1])[1]],"speed": 999 ,"lane": 999 ,"timestamp":int(time.time())})

#############################3
############################3
###########################3
                        if i == 2:
                            # cctv_traffic.append({"traffic":{"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS_ch3(output[0],output[1])[0], toWGS_ch3(output[0],output[1])[1]],"speed": None ,"lane": None ,"timestamp":int(time.time())}})
                            cctv_traffic.append({"cctv":i+1,"class":int(output[5]),"id":id,"bbox":[output[0],output[1],output[2],output[3]],"GPS":[toWGS_ch3(output[0],output[1])[0], toWGS_ch3(output[0],output[1])[1]],"speed": 999 ,"lane": 999 ,"timestamp":int(time.time())})
#######################################3
############################3
############################3
#################3


                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                        if i == 1:
                            AV_eta = 0

                            
                            if bbox_y > 80:
                                # ch02_exist = True
                                if save_vid or save_crop or show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id

                                    label =   (f'{names[c]} {int(abs(round_to_multiple(final_speed,2)))}km/h')






                                    # label =   (f'{names[c]} {int(abs(round_to_multiple(final_speed,2)))}km/h')



                                    # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    # label =   (f'{id} {names[c]} {conf:.2f} {int(abs(round_to_multiple(final_speed,2)))}km/h {bbox_x} {bbox_y}')
                                    # label =   (f'{names[c]} {int(abs(round_to_multiple(final_speed,2)))}km/h')
                                    # label =   (f'{id} {names[c]}  {int(abs(round_to_multiple(final_speed,2)))}km/h')
                                    # print(label)
                                        # (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f} {final_speed}'))
                                    if id == 88888:
                                        if len(output)>6:
                                            if eta_flag:
                                                AV_eta = round(eta(toWGS(output[8],output[9]),output[7]),2)
                                                if AV_eta < 2.10:
                                                    eta_flag = False

                                        # if ch01_exist:
                                        #     c = 19
                                        # else:
                                        c = 50
                                        label =   (f'AV White {int(abs(round_to_multiple(final_speed,2)))}km/h')
                                        AV_flag = True
                                        


                                    # if ch01_exist:
                                    #     c = 19


                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                    if save_crop:
                                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        elif i ==0:
                            if save_vid or save_crop or show_vid:  # Add bbox to image

                                c = int(cls)

                                id = int(id)  # integer id
                                # '''
                                # label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                #     (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                # '''
                                # label = None if hide_labels else (f'{id}' if hide_conf else \
                                #     # (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                #     (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]}'))
                                label =   (f'{names[c]}')


# ##########################3
#                                 for zoom_idx, zoom_val in enumerate(zoom_av_candidate):
#                                     check_id = int(id)
#                                     zoom_id = int(zoom_val)
#                                     print('color')
#                                     print(id)
#                                     print(zoom_av_candidate)
#                                     print(zoom_color)
#                                     if check_id == zoom_val:
#                                         car_color = zoom_color[zoom_idx]
#                                         label =   (f'{names[c]} {car_color}')
###############################3


                                # AV_eta = 0 
                                # AV_eta = round(eta(toWGS(output[8],output[9]),output[7]),2)


                                if id == 88888:
                                    c = 50
                                # print(bboxes)
                                # print(label)
                                    label =   (f'AV White')


                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)


                        elif i ==2:
                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                # if id == 88888:
                                # id = int(id)

                                id = int(id)  # integer id
                                c = int(cls)
                                # label = None if hide_labels else (f'{id}' if hide_conf else \
                                #     # (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                #     (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]}'))
                                label =   (f'{names[c]}')




                                # if ch02_exist:
                                #     c = 50
                                # else:
                                #     c = 50
                                #     c = int(cls)

                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)


                with open(cctv_traffic_json, "w", encoding='utf-8') as json_file:
                    print('cctv traffic')
                    print(cctv_traffic)
                    cctv_traffic_dict = {"traffic":cctv_traffic}
                    # cctv_traffic_dict = {"traffic":[]}
                    # cctv_traffic_dict["traffic"].append(cctv_traffic)
                    print(cctv_traffic_dict)
                    json.dump(cctv_traffic_dict, json_file, indent=4)

                    
###########################################
                # with open('json_sample/cctv_traffic'+str(frame_idx)+'.json', "w", encoding='utf-8') as json_file:
                #     # print(cctv_traffic_vds)
                #     # json.dump(cctv_traffic_dict, json_file, indent=4)
                #     print('cctv traffic')
                #     print(cctv_traffic)
                #     cctv_traffic_dict = {"traffic":cctv_traffic}
                #     # cctv_traffic_dict = {"traffic":[]}
                #     # cctv_traffic_dict["traffic"].append(cctv_traffic)
                #     print(cctv_traffic_dict)
                #     json.dump(cctv_traffic_dict, json_file, indent=4)
###########################################33





                with open(entry_json, "w", encoding='utf-8') as json_file:
                    print('entry')
                    print(entry)
                    json.dump(entry, json_file, indent=4)

                    
# ###########################################
#                 with open('json_sample/entry'+str(frame_idx)+'.json', "w", encoding='utf-8') as json_file:
#                     # print(cctv_traffic_vds)
#                     json.dump(entry, json_file, indent=4)
###########################################



                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()

            if i ==2 :

                # cv2.putText(frame, "Current Traffic: "+str(traffic_count),(int(20), int(80)),0, 5e-3 * 200, (0,255,0), 2)


                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time()).isoformat()), (540,40), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time())), (540,80), 0, 1, (0,255,0), 2)
                
                


                if ch02_exist:

                    cv2.putText(im0, "Status: Warning", (50,40), 0, 1, (0,0,255), 2)
                else:
                    cv2.putText(im0, "Status: Safe", (50,40), 0, 1, (255,255,255), 2)
              

            if i == 0 :

                # cv2.putText(frame, "Current Traffic: "+str(traffic_count),(int(20), int(80)),0, 5e-3 * 200, (0,255,0), 2)


                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time()).isoformat()), (540,40), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time())), (540,80), 0, 1, (0,255,0), 2)
                
                

                print('zoom candidate')
                print(zoom_av_candidate)

                # if av_car_flag:
                if 200.0 in zoom_av_candidate:

                    
                    cv2.putText(im0, "AV ID: Identified", (50,40), 0, 1, (255,0,0), 2)
                    # cv2.putText(im0, "ETA: "+ str(eta(toWGS(output[8],output[9]))), (50,80), 0, 1, (255,255,255), 2)
                    # if ch01_exist:
                    #     cv2.putText(im0, "Status: Warning", (50,120), 0, 1, (255,0,0), 2)
                    # else:
                    #     cv2.putText(im0, "Status: Good", (50,120), 0, 1, (255,255,255), 2)
                else:
                    cv2.putText(im0, "AV ID: Not Identified", (50,40), 0, 1, (255,255,255), 2)
                    # cv2.putText(im0, "Junction ETA: "+ str(eta(toWGS(output[8],output[9]))), (50,80), 0, 1, (255,255,255), 2)
                    # if ch01_exist:
                    #     cv2.putText(im0, "Status: Warning", (50,120), 0, 1, (255,0,0), 2)
                    # else:
                    #     cv2.putText(im0, "Status: Good", (50,120), 0, 1, (255,255,255), 2)

            if i == 1 :

                # cv2.putText(frame, "Current Traffic: "+str(traffic_count),(int(20), int(80)),0, 5e-3 * 200, (0,255,0), 2)


                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time()).isoformat()), (540,40), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Time "+str(datetime.datetime.fromtimestamp(time.time())), (540,80), 0, 1, (0,255,0), 2)
                
                


                if AV_flag:
                # if 88888 in zoom_av_candidate:
                    cv2.putText(im0, "AV Re-ID: Re-Identified", (50,80), 0, 1, (255,255,0), 2)
                    cv2.putText(im0, "Junction ETA: "+ str(AV_eta)+'s', (50,120), 0, 1, (255,255,255), 2)
                    cv2.putText(im0, "AV Lane: 3", (50,160), 0, 1, (255,255,255), 2)
                    # if len(output)>6:
                    #     cv2.putText(im0, "Junction ETA: "+ str(AV_eta), (50,80), 0, 1, (255,255,255), 2)
                    # else:
                    #     cv2.putText(im0, "Junction ETA: NA", (50,80), 0, 1, (255,255,255), 2)

                    if ch01_exist:
                        cv2.putText(im0, "Status: Warning", (50,40), 0, 1, (0,0,255), 2)
                    else:
                        cv2.putText(im0, "Status: Safe", (50,40), 0, 1, (255,255,255), 2)
                else:
                    cv2.putText(im0, "AV Re-ID: Not Identified", (50,80), 0, 1, (255,255,255), 2)
                    cv2.putText(im0, "Junction ETA: "+ str(AV_eta)+'s', (50,120), 0, 1, (255,255,255), 2)
                    cv2.putText(im0, "AV Lane: NA", (50,160), 0, 1, (255,255,255), 2)
                    # if len(output)>6:
                    #     cv2.putText(im0, "Junction ETA: "+ str(AV_eta), (50,80), 0, 1, (255,255,255), 2)
                    # else:
                    #     cv2.putText(im0, "Junction ETA: NA", (50,80), 0, 1, (255,255,255), 2)
                    if ch01_exist:
                        cv2.putText(im0, "Status: Warning", (50,40), 0, 1, (0,0,255), 2)
                    else:
                        cv2.putText(im0, "Status: Safe", (50,40), 0, 1, (255,255,255), 2)

                # cv2.putText(im0, "Lane #: Traffic, Speed", (540,40), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Lane 1: " + str(len(traffic_lane_1))+', '+str(int(avg_speed_lane_1)), (540,80), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Lane 2: " + str(len(traffic_lane_2))+', '+str(int(avg_speed_lane_2)), (540,120), 0, 1, (0,255,0), 2)
                # cv2.putText(im0, "Lane 3: " + str(len(traffic_lane_3))+', '+str(int(avg_speed_lane_3)), (540,160), 0, 1, (0,255,0), 2)
                
                
                # cv2.circle(im0, (int((bboxes[2] + bboxes[0])/ 2),int(bboxes[3] )), 0, color=(0,0,255), thickness=10)
                # cv2.circle(im0, (bot_x,bot_y), 0, color=(0,0,255), thickness=10)
                            # center_x = (bboxes[2] + bboxes[0])/ 2
                            # center_y = (bboxes[3] + bboxes[1])/ 2
                

                lane1_speed, lane1_count = calc_lane_speed(speed_lane_1)
                lane2_speed, lane2_count = calc_lane_speed(speed_lane_2)
                lane3_speed, lane3_count = calc_lane_speed(speed_lane_3)
                
                cctv_traffic_vds.append({"cctv":i+1,"lane1_speed": lane1_speed,"lane1_count": lane1_count,"lane2_speed": lane2_speed,"lane2_count": lane2_count,"lane3_speed": lane3_speed,"lane3_count": lane3_count,"merge_count": int(merge_count),"timestamp":int(time.time())})
                print('vds')
                print(cctv_traffic_vds)
                with open(cctv_traffic_vds_json, "w", encoding='utf-8') as json_file:
                    # print(cctv_traffic_vds)
                    json.dump(cctv_traffic_vds, json_file, indent=4)


###########################
                # with open('json_sample/cctv_traffic_vds+'+str(frame_idx)+'.json', "w", encoding='utf-8') as json_file:
                #     # print(cctv_traffic_vds)
                #     json.dump(cctv_traffic_vds, json_file, indent=4)

###################################


                if frame_index % 150 == 0:  
                    speed_lane_1.clear()
                    speed_lane_2.clear()
                    speed_lane_3.clear()    
                    traffic_lane_1.clear()
                    traffic_lane_2.clear()
                    traffic_lane_3.clear()


            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid :
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path

                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

            # if  frame_idx == 1000:
            #     vid_writer[i].release()
            #     print('##############################################################')
            #     print('##############################################################')
            #     print('##############################################################')
            #     print('##############################################################')
        
        f_end = time.time()

        file_name = os.path.basename(source)
        time_per_frame = f_end - f_start

        num_track_z = 0
        num_track_n = len(tracker_list[1].tracker.tracks)
        num_track_1 = 0

        num_obj_z = len(pred_z[0])
        num_obj_n = len(pred[0])
        num_obj_1 = len(pred_1[0])

        time_detection = t3 - t2
        time_tracking = track_time
        time_reid = reid_time
        print('time per a frame :', time_per_frame)

    os.makedirs('benchmark/v8', exist_ok=True)
    result_path = 'benchmark/v8/' + file_name[:-4] + '_result' + '.csv'
    

     
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'best_20221215.engine', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x1_0_vehicle.engine')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='test.mp4', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

import geopy
# def convPtoR(x1,y1,x2,y2) :
#     minus_flag = False
#     H = [[ 7.41461933e-03 ,-5.63196412e-01,  1.26611368e+02],
#  [ 2.19803547e-03 ,-1.66933111e-01 , 3.75276050e+01],
#  [ 5.85638168e-05 ,-4.44821357e-03,  1.00000000e+00]]
#     B1 = np.array([x1,y1,1])
#     R1 = np.matmul(H,B1)
#     B2 = np.array([x2,y2,1])
#     R2 = np.matmul(H,B2)
#     R1 /= R1[2]
#     R2 /= R2[2]
    
#     x_distance = ((R1[0]-R2[0]) * 100000 *1.1 ) #보정
#     y_distance = ((R1[1]-R2[1]) * 100000 *0.9)
#     if y_distance < 0:
#         minus_flag = True

#     # R_dist = math.sqrt(x_distance*x_distance + y_distance*y_distance)


#     R_dist = geopydist.geodesic((R1[1],R1[0]), (R2[1],R2[0])).meters
#     return R_dist, minus_flag




def eta(xy,speed) :
    minus_flag = False
#     H = [[ 7.41461933e-03 ,-5.63196412e-01,  1.26611368e+02],
#  [ 2.19803547e-03 ,-1.66933111e-01 , 3.75276050e+01],
#  [ 5.85638168e-05 ,-4.44821357e-03,  1.00000000e+00]]
#     B1 = np.array([x1,y1,1])
#     R1 = np.matmul(H,B1)
#     R1 /= R1[2]


    R1 = xy
    R2 = [126.6117963,37.52807626]
    
    # x_distance = ((R1[0]-R2[0]) * 100000  ) #보정
    # y_distance = ((R1[1]-R2[1]) * 100000 )


    # print(R1)
    # print(R2)


    R_dist = geopydist.geodesic((R1[0],R1[1]), (R2[1],R2[0])).meters
    if speed ==0:
        eta1 = 0
    else:
        eta1 = R_dist /1000 / speed * 3600
    return eta1


def toWGS(x1,y1) :
    minus_flag = False
    H = [[-9.78543003e-03,2.87021092e-01,  3.75291743e+01],
 [-3.30121073e-02,  9.68347160e-01,  1.26612531e+02],
 [-2.60738145e-04,  7.64815819e-03,  1.00000000e+00]]
    B1 = np.array([x1,y1,1])
    R1 = np.matmul(H,B1)

    R1 /= R1[2]

    return R1

def toWGS_ch1(x1,y1) :
    minus_flag = False
    H = [[-1.73196607e-02,  1.72775174e-01,  3.75277200e+01],
 [-5.84338451e-02 , 5.82905924e-01,  1.26611401e+02],
 [-4.61518409e-04 , 4.60388088e-03 , 1.00000000e+00]]
    B1 = np.array([x1,y1,1])
    R1 = np.matmul(H,B1)

    R1 /= R1[2]

    return R1

def toWGS_ch3(x1,y1) :
    minus_flag = False
    H = [[ 1.91337616e-03 ,-1.82463459e-01,  3.75270836e+01],
 [ 6.45467909e-03 ,-6.15591084e-01  ,1.26611127e+02],
 [ 5.09831519e-05 ,-4.86202848e-03  ,1.00000000e+00]]
    B1 = np.array([x1,y1,1])
    R1 = np.matmul(H,B1)

    R1 /= R1[2]

    return R1

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
