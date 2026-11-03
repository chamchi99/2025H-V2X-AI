import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from sort.nn_matching import NearestNeighborDistanceMetric
from sort.detection import Detection
from sort.tracker_comb import Tracker

from reid_multibackend import ReIDDetectMultiBackend

from yolov5.utils.general import xyxy2xywh

from calc_speed_sort_comb import *
from ch_setting_comb_sort import *
from PIL import Image



class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds)

        self.id_list = []
        self.bot_list = []
        self.frame_list = []


        # self.lane_1 = [] 
        # self.lane_2 = []
        # self.lane_3 = []


        # self.mask = []
        # self.global_point = []

        # self.speed_lane_1 = []
        # self.speed_lane_2 = []
        # self.speed_lane_3 = []


        # self.traffic_lane_1 = []
        # self.traffic_lane_2 = []
        # self.traffic_lane_3 = []




    # def update(self, dets,  ori_img, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_list,frame_index):
    def update(self, dets,  ori_img,lane_1, lane_2, lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane,mask, frame_index):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)
        
        # output bbox identities
        outputs = []
        traffic_id = []
        for track in self.tracker.tracks:

            # if count_box == 0 and len(tracker.tracks) > 0:
            #     tracker.tracks.clear()


            if not track.is_confirmed() or track.time_since_update > 1:

                if track.age > track._max_age:
                    track.state = 3       
                # print('a')
                continue
            track.pop_flag = True

            # traffic_id.append(int(track.track_id))

            traffic_id.append(int(track.track_id))
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            box = self._tlwh_to_xyxy(box)
            class_name = track.get_class()

##################################################################################

            bbox_center = (int(((box[0])+(box[2]))/2), int(box[3]))
            right_bottom = (int(box[2]), int(box[3]))
            left_up = (int(box[0]), int(box[1]))

            # mask ROI filter removed - detect all visible objects
            # if cv2.pointPolygonTest(mask, (int(bbox_center[0]),bbox_center[1]) , False) < 0:
            #     continue



            # print(box)
            # print(bbox_center)
            # print(x1,y1,x2,y2)

            # print(bbox_center[1])
            if bbox_center[1] > 380:
                # print(bbox_center[1])
                track.matching_point[0] = bbox_center[0]
                track.matching_point[1] = int((box[1] + box[3]) / 2)  # [실험] foot(box[3])→box 중심 y
                track.matching_point[2] = frame_index   
                track.point_y.append(bbox_center[1])


                if bbox_center[1] >570:

                    if cv2.pointPolygonTest(lane_1, (int(bbox_center[0]*1.2),bbox_center[1]) , False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    elif cv2.pointPolygonTest(lane_2, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    elif cv2.pointPolygonTest(lane_3, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                else :

                    if cv2.pointPolygonTest(lane_1, bbox_center, False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    elif cv2.pointPolygonTest(lane_2, bbox_center, False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    elif cv2.pointPolygonTest(lane_3, bbox_center, False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                speed_list = [track.matching_point[0], track.matching_point[1], frame_index]
                channel = '08'
                # print(avg_speed_lane)
                # print(speed_list)
                sspeed = CalcSpeed(track, convPtoR, speed_list, channel, avg_speed_lane)
                # print(sspeed)
                speed_interval = 2

                # if track.speed_valid:
                #     if frame_index % speed_interval == 0:   

                #         if sspeed == 0:
                #             track.speed = sspeed
                #         else: 
                #             #update tracker speed attribute
                #             if len(track.eval_speed) != 0:
                #                 if track.eval_speed[-1] != 0 and track.eval_speed[-1] > 20:
                #                     if sspeed > 1.1*track.eval_speed[-1]:
                #                         track.speed = int(1.1*track.eval_speed[-1])
                #                     elif sspeed < 0.9*track.eval_speed[-1]:
                #                         track.speed = int(0.9*track.eval_speed[-1])
                #                     else:
                #                         track.speed = sspeed
                #                 else:
                #                     track.speed = sspeed
                #             else:
                #                 track.speed = sspeed

                #             if avg_speed_lane[track.driving_lane-1] != -999  and \
                #                 avg_speed_lane[track.driving_lane-1] != 0 and \
                #                     avg_speed_lane[track.driving_lane-1]>50:
                #                 if track.speed > 1.5*avg_speed_lane[track.driving_lane-1]:
                #                     track.speed = int(1.2*avg_speed_lane[track.driving_lane-1])
                                        
                #         if -4 <= track.speed <=-1 or 1 <= track.speed <= 4:
                #             track.speed = 0
                #         ##########################
                #         track.eval_speed.append(track.speed)



                if track.driving_lane != 0:


                    if track.driving_lane == 1 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_1.append(track.speed)
                    elif track.driving_lane == 2 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_2.append(track.speed)
                    elif track.driving_lane == 3 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_3.append(track.speed)

##################################################################################




            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            speed_final = track.speed
            # speed_final = sspeed
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf,speed_final,track.matching_point[0],track.matching_point[1],track.driving_lane]))
            # print(class_id)
        avg_speed_lane_1 = sum(speed_lane_1, 0.0) / len(speed_lane_1) if len(speed_lane_1) != 0 else -999
        avg_speed_lane_2 = sum(speed_lane_2, 0.0) / len(speed_lane_2) if len(speed_lane_2) != 0 else -999
        avg_speed_lane_3 = sum(speed_lane_3, 0.0) / len(speed_lane_3) if len(speed_lane_3) != 0 else -999
        avg_speed_lane = [avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3]


        # print(outputs)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        # return outputs, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 
        return outputs,lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 ,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def trajectory(self, im0, tracks, color, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        for t in tracks:
            for p in t.q:
                if p[0] is 'observationupdate': 
                    cv2.circle(im0, p[1], 2, color, 2)
                else:
                    cv2.circle(im0, p[1], 2, (255,255,255), 2)












class StrongSORT_w_Matching(object):
    def __init__(self,
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9,
                 query_img_path=None
                 ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        self.flag = False  # whether self-driving car is determined
        query_img = [np.asarray(Image.open(query_img_path))]
        self.matching_thres = 0.28
        query_feature = self.model(query_img).detach().cpu().numpy()
        self.query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)



    def update(self, dets, ori_img):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # classify self-driving car
        if not self.flag:
            img_crops = []
            track_candidate = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                cen_x = (x1 + x2) / 2
                cen_y = (y1 + y2) / 2

                if (500 < cen_y) and (cen_y < 750):
                    if (372 < cen_x) and (cen_x < 1688):
                        im = ori_img[y1:y2, x1:x2]
                        img_crops.append(im)
                        track_candidate.append(track)

            if img_crops:
                features = self.model(img_crops).detach().cpu().numpy()
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                dist = 1. - np.dot(self.query_feature, features.T)

                for i in range(np.shape(dist)[1]):
                    if dist[0, i] < self.matching_thres:
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')
                        print('success to classify self-driving car')
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')

                        track_candidate[i].is_AV = True
                        self.flag = True

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            if track.is_AV:
                track_id = 88888

            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features






















class StrongSORT1(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds)

        self.id_list = []
        self.bot_list = []
        self.frame_list = []




    # def update(self, dets,  ori_img, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_list,frame_index):
    def update(self, dets, ori_img):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:

            # if count_box == 0 and len(tracker.tracks) > 0:
            #     tracker.tracks.clear()


            if not track.is_confirmed() or track.time_since_update > 1:

                continue
            track.pop_flag = True

            # traffic_id.append(int(track.track_id))

            # traffic_id.append(int(track.track_id))
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            box = self._tlwh_to_xyxy(box)
            class_name = track.get_class()

##################################################################################

            bbox_center = (int(((box[0])+(box[2]))/2), int(box[3]))




            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            speed_final = track.speed
            # speed_final = sspeed
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
            # print(class_id)


        # print(outputs)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        # return outputs, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 
        return outputs
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def trajectory(self, im0, tracks, color, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        for t in tracks:
            for p in t.q:
                if p[0] is 'observationupdate': 
                    cv2.circle(im0, p[1], 2, color, 2)
                else:
                    cv2.circle(im0, p[1], 2, (255,255,255), 2)








class StrongSORT_w_v_and_m(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9,
                 query_img_path=None
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds)

        self.id_list = []
        self.bot_list = []
        self.frame_list = []

        self.flag = False  # whether self-driving car is determined
        query_img = [np.asarray(Image.open(query_img_path))]
        self.matching_thres = 0.28
        query_feature = self.model(query_img).detach().cpu().numpy()
        self.query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)


        # self.lane_1 = [] 
        # self.lane_2 = []
        # self.lane_3 = []


        # self.mask = []
        # self.global_point = []

        # self.speed_lane_1 = []
        # self.speed_lane_2 = []
        # self.speed_lane_3 = []


        # self.traffic_lane_1 = []
        # self.traffic_lane_2 = []
        # self.traffic_lane_3 = []




    # def update(self, dets,  ori_img, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_list,frame_index):
    def update(self, dets,  ori_img,lane_1, lane_2, lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane,mask, frame_index):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # classify self-driving car
        if not self.flag:
            img_crops = []
            track_candidate = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                cen_x = (x1 + x2) / 2
                cen_y = (y1 + y2) / 2

                if (450 < cen_y) and (cen_y < 800):
                    if (20 < cen_x) and (cen_x < 400):
                        im = ori_img[y1:y2, x1:x2]
                        img_crops.append(im)
                        track_candidate.append(track)

            if img_crops:
                features = self.model(img_crops).detach().cpu().numpy()
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                dist = 1. - np.dot(self.query_feature, features.T)

                for i in range(np.shape(dist)[1]):
                    if dist[0, i] < self.matching_thres:
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')
                        print('success to classify self-driving car')
                        print('************************************************')
                        print('************************************************')
                        print('************************************************')

                        track_candidate[i].is_AV = True
                        self.flag = True

        # output bbox identities
        outputs = []
        traffic_id = []
        for track in self.tracker.tracks:

            # if count_box == 0 and len(tracker.tracks) > 0:
            #     tracker.tracks.clear()


            if not track.is_confirmed() or track.time_since_update > 1:

                if track.age > track._max_age:
                    track.state = 3       
                # print('a')
                continue
            track.pop_flag = True

            # traffic_id.append(int(track.track_id))

            traffic_id.append(int(track.track_id))
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            box = self._tlwh_to_xyxy(box)
            class_name = track.get_class()

##################################################################################

            bbox_center = (int(((box[0])+(box[2]))/2), int(box[3]))
            right_bottom = (int(box[2]), int(box[3]))
            left_up = (int(box[0]), int(box[1]))

            # mask ROI filter removed - detect all visible objects
            # if cv2.pointPolygonTest(mask, (int(bbox_center[0]),bbox_center[1]) , False) < 0:
            #     continue



            # print(box)
            # print(bbox_center)
            # print(x1,y1,x2,y2)

            # print(bbox_center[1])
            if bbox_center[1] > 380:
                # print(bbox_center[1])
                track.matching_point[0] = bbox_center[0]
                track.matching_point[1] = int((box[1] + box[3]) / 2)  # [실험] foot(box[3])→box 중심 y
                track.matching_point[2] = frame_index   
                track.point_y.append(bbox_center[1])


                if bbox_center[1] >570:

                    if cv2.pointPolygonTest(lane_1, (int(bbox_center[0]*1.2),bbox_center[1]) , False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    elif cv2.pointPolygonTest(lane_2, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    elif cv2.pointPolygonTest(lane_3, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                else :

                    if cv2.pointPolygonTest(lane_1, bbox_center, False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    elif cv2.pointPolygonTest(lane_2, bbox_center, False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    elif cv2.pointPolygonTest(lane_3, bbox_center, False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                speed_list = [track.matching_point[0], track.matching_point[1], frame_index]
                channel = '08'
                # print(avg_speed_lane)
                # print(speed_list)
                sspeed = CalcSpeed(track, convPtoR, speed_list, channel, avg_speed_lane)
                # print(sspeed)
                speed_interval = 2

                # if track.speed_valid:
                #     if frame_index % speed_interval == 0:   

                #         if sspeed == 0:
                #             track.speed = sspeed
                #         else: 
                #             #update tracker speed attribute
                #             if len(track.eval_speed) != 0:
                #                 if track.eval_speed[-1] != 0 and track.eval_speed[-1] > 20:
                #                     if sspeed > 1.1*track.eval_speed[-1]:
                #                         track.speed = int(1.1*track.eval_speed[-1])
                #                     elif sspeed < 0.9*track.eval_speed[-1]:
                #                         track.speed = int(0.9*track.eval_speed[-1])
                #                     else:
                #                         track.speed = sspeed
                #                 else:
                #                     track.speed = sspeed
                #             else:
                #                 track.speed = sspeed

                #             if avg_speed_lane[track.driving_lane-1] != -999  and \
                #                 avg_speed_lane[track.driving_lane-1] != 0 and \
                #                     avg_speed_lane[track.driving_lane-1]>50:
                #                 if track.speed > 1.5*avg_speed_lane[track.driving_lane-1]:
                #                     track.speed = int(1.2*avg_speed_lane[track.driving_lane-1])
                                        
                #         if -4 <= track.speed <=-1 or 1 <= track.speed <= 4:
                #             track.speed = 0
                #         ##########################
                #         track.eval_speed.append(track.speed)



                if track.driving_lane != 0:


                    if track.driving_lane == 1 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_1.append(track.speed)
                    elif track.driving_lane == 2 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_2.append(track.speed)
                    elif track.driving_lane == 3 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_3.append(track.speed)

##################################################################################




            track_id = track.track_id
            if track.is_AV:
                track_id = 88888
            class_id = track.class_id
            conf = track.conf
            speed_final = track.speed
            # speed_final = sspeed
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf,speed_final,track.matching_point[0],track.matching_point[1],track.driving_lane]))
            # print(class_id)
        avg_speed_lane_1 = sum(speed_lane_1, 0.0) / len(speed_lane_1) if len(speed_lane_1) != 0 else -999
        avg_speed_lane_2 = sum(speed_lane_2, 0.0) / len(speed_lane_2) if len(speed_lane_2) != 0 else -999
        avg_speed_lane_3 = sum(speed_lane_3, 0.0) / len(speed_lane_3) if len(speed_lane_3) != 0 else -999
        avg_speed_lane = [avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3]


        # print(outputs)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        # return outputs, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 
        return outputs,lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 ,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def trajectory(self, im0, tracks, color, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        for t in tracks:
            for p in t.q:
                if p[0] is 'observationupdate': 
                    cv2.circle(im0, p[1], 2, color, 2)
                else:
                    cv2.circle(im0, p[1], 2, (255,255,255), 2)









                    
################################################

class StrongSORT_rotate(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds)

        self.id_list = []
        self.bot_list = []
        self.frame_list = []


        # self.lane_1 = [] 
        # self.lane_2 = []
        # self.lane_3 = []


        # self.mask = []
        # self.global_point = []

        # self.speed_lane_1 = []
        # self.speed_lane_2 = []
        # self.speed_lane_3 = []


        # self.traffic_lane_1 = []
        # self.traffic_lane_2 = []
        # self.traffic_lane_3 = []




    # def update(self, dets,  ori_img, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_list,frame_index):
    def update(self, dets,  ori_img,lane_1, lane_2, lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane,mask, frame_index):

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)
        
        # output bbox identities
        outputs = []
        traffic_id = []
        for track in self.tracker.tracks:

            # if count_box == 0 and len(tracker.tracks) > 0:
            #     tracker.tracks.clear()


            if not track.is_confirmed() or track.time_since_update > 1:

                if track.age > track._max_age:
                    track.state = 3       
                # print('a')
                continue
            track.pop_flag = True

            # traffic_id.append(int(track.track_id))

            traffic_id.append(int(track.track_id))
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            box = self._tlwh_to_xyxy(box)
            class_name = track.get_class()

##################################################################################

            bbox_center = (int(((box[0])+(box[2]))/2), int(box[3]))
            right_bottom = (int(box[2]), int(box[3]))
            left_up = (int(box[0]), int(box[1]))

            # mask 영역 외 track은 output에서 제외 (보행자/차량 모두 포함)
            if len(mask) >= 3 and cv2.pointPolygonTest(mask, bbox_center, False) < 0:
                continue

            # print(box)
            # print(bbox_center)
            # print(x1,y1,x2,y2)

            # print(bbox_center[1])
            if bbox_center[1] > 100:
                # print(bbox_center[1])
                track.matching_point[0] = bbox_center[0]
                track.matching_point[1] = int((box[1] + box[3]) / 2)  # [실험] foot(box[3])→box 중심 y
                track.matching_point[2] = frame_index   
                track.point_y.append(bbox_center[1])


                if bbox_center[1] >120:

                    # if cv2.pointPolygonTest(lane_1, (int(bbox_center[0]*1.2),bbox_center[1]) , False) >= 0:
                    if cv2.pointPolygonTest(lane_1, (int(bbox_center[0]),bbox_center[1]) , False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    # elif cv2.pointPolygonTest(lane_2, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                    elif cv2.pointPolygonTest(lane_2, (int(bbox_center[0]),bbox_center[1]), False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    # elif cv2.pointPolygonTest(lane_3, (int(bbox_center[0]*1.2),bbox_center[1]), False) >= 0:
                    elif cv2.pointPolygonTest(lane_3, (int(bbox_center[0]),bbox_center[1]), False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                else :

                    if cv2.pointPolygonTest(lane_1, bbox_center, False) >= 0:
                        track.driving_lane = 1
                        if track.track_id not in traffic_lane_1 :
                            traffic_lane_1.append(track.track_id)


                    elif cv2.pointPolygonTest(lane_2, bbox_center, False) >= 0:
                        track.driving_lane = 2
                        if track.track_id not in traffic_lane_2 :
                                traffic_lane_2.append(track.track_id)

                    elif cv2.pointPolygonTest(lane_3, bbox_center, False) >= 0:
                        track.driving_lane = 3
                        if track.track_id not in traffic_lane_3 :
                                traffic_lane_3.append(track.track_id)

                speed_list = [track.matching_point[0], track.matching_point[1], frame_index]
                channel = '08'
                # print(avg_speed_lane)
                # print(speed_list)
                sspeed = CalcSpeed_rotate(track, convPtoR_rotate, speed_list, channel, avg_speed_lane)
                # print(sspeed)
                speed_interval = 2

                # if track.speed_valid:
                #     if frame_index % speed_interval == 0:   

                #         if sspeed == 0:
                #             track.speed = sspeed
                #         else: 
                #             #update tracker speed attribute
                #             if len(track.eval_speed) != 0:
                #                 if track.eval_speed[-1] != 0 and track.eval_speed[-1] > 20:
                #                     if sspeed > 1.1*track.eval_speed[-1]:
                #                         track.speed = int(1.1*track.eval_speed[-1])
                #                     elif sspeed < 0.9*track.eval_speed[-1]:
                #                         track.speed = int(0.9*track.eval_speed[-1])
                #                     else:
                #                         track.speed = sspeed
                #                 else:
                #                     track.speed = sspeed
                #             else:
                #                 track.speed = sspeed

                #             if avg_speed_lane[track.driving_lane-1] != -999  and \
                #                 avg_speed_lane[track.driving_lane-1] != 0 and \
                #                     avg_speed_lane[track.driving_lane-1]>50:
                #                 if track.speed > 1.5*avg_speed_lane[track.driving_lane-1]:
                #                     track.speed = int(1.2*avg_speed_lane[track.driving_lane-1])
                                        
                #         if -4 <= track.speed <=-1 or 1 <= track.speed <= 4:
                #             track.speed = 0
                #         ##########################
                #         track.eval_speed.append(track.speed)



                if track.driving_lane != 0:


                    if track.driving_lane == 1 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_1.append(track.speed)
                    elif track.driving_lane == 2 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_2.append(track.speed)
                    elif track.driving_lane == 3 and track.speed >= 0 and track.speed <= 200:
                        speed_lane_3.append(track.speed)

##################################################################################




            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            speed_final = track.speed
            # speed_final = sspeed
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf,speed_final,track.matching_point[0],track.matching_point[1],track.driving_lane,bbox_center[0],bbox_center[1]]))
            # print(class_id)
        avg_speed_lane_1 = sum(speed_lane_1, 0.0) / len(speed_lane_1) if len(speed_lane_1) != 0 else -999
        avg_speed_lane_2 = sum(speed_lane_2, 0.0) / len(speed_lane_2) if len(speed_lane_2) != 0 else -999
        avg_speed_lane_3 = sum(speed_lane_3, 0.0) / len(speed_lane_3) if len(speed_lane_3) != 0 else -999
        avg_speed_lane = [avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3]


        # print(outputs)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        # return outputs, lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 
        return outputs,lane_1,lane_2,lane_3,speed_lane_1,speed_lane_2,speed_lane_3,traffic_lane_1,traffic_lane_2,traffic_lane_3 ,avg_speed_lane_1,avg_speed_lane_2,avg_speed_lane_3,avg_speed_lane
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def trajectory(self, im0, tracks, color, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        for t in tracks:
            for p in t.q:
                if p[0] is 'observationupdate': 
                    cv2.circle(im0, p[1], 2, color, 2)
                else:
                    cv2.circle(im0, p[1], 2, (255,255,255), 2)




