from typing import List, Tuple, Union

import numpy as np
from config import ModelType
from numpy import ndarray


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


class Decoder:

    def __init__(self, model_type: ModelType, model_only: bool = False):
        self.model_type = model_type
        self.model_only = model_only
        self.boxes_pro = []
        self.scores_pro = []
        self.labels_pro = []
        self.is_logging = False

    def __call__(self,
                 feats: Union[List, Tuple],
                 conf_thres: float,
                 num_labels: int = 80,
                 **kwargs) -> Tuple:
        if not self.is_logging:
            print('Only support decode in batch==1')
            self.is_logging = True
        self.boxes_pro.clear()
        self.scores_pro.clear()
        self.labels_pro.clear()

        if self.model_only:
            # transpose channel to last dim for easy decoding
            feats = [
                np.ascontiguousarray(feat[0].transpose(1, 2, 0))
                for feat in feats
            ]
        else:
            # ax620a horizonX3 transpose channel to last dim by default
            feats = [np.ascontiguousarray(feat) for feat in feats]
        if self.model_type == ModelType.YOLOV5:
            self.__yolov5_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == ModelType.YOLOX:
            self.__yolox_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type in (ModelType.PPYOLOE, ModelType.PPYOLOEP):
            self.__ppyoloe_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == ModelType.YOLOV6:
            self.__yolov6_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == ModelType.YOLOV7:
            self.__yolov7_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == ModelType.RTMDET:
            self.__rtmdet_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == ModelType.YOLOV8:
            self.__yolov8_decode(feats, conf_thres, num_labels, **kwargs)
        else:
            raise NotImplementedError
        return self.boxes_pro, self.scores_pro, self.labels_pro

    def __yolov5_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        anchors: Union[List, Tuple] = kwargs.get(
            'anchors',
            [[(10, 13), (16, 30),
              (33, 23)], [(30, 61), (62, 45),
                          (59, 119)], [(116, 90), (156, 198), (373, 326)]])
        for i, feat in enumerate(feats):
            stride = 8 << i
            feat_h, feat_w, _ = feat.shape
            anchor = anchors[i]
            feat = sigmoid(feat)
            feat = feat.reshape((feat_h, feat_w, len(anchor), -1))
            box_feat, conf_feat, score_feat = np.split(feat, [4, 5], -1)

            hIdx, wIdx, aIdx, _ = np.where(conf_feat > conf_thres)

            num_proposal = hIdx.size
            if not num_proposal:
                continue

            score_feat = score_feat[hIdx, wIdx, aIdx] * conf_feat[hIdx, wIdx,
                                                                  aIdx]
            boxes = box_feat[hIdx, wIdx, aIdx]
            labels = score_feat.argmax(-1)
            scores = score_feat.max(-1)

            indices = np.where(scores > conf_thres)[0]
            if len(indices) == 0:
                continue

            for idx in indices:
                a_w, a_h = anchor[aIdx[idx]]
                x, y, w, h = boxes[idx]
                x = (x * 2.0 - 0.5 + wIdx[idx]) * stride
                y = (y * 2.0 - 0.5 + hIdx[idx]) * stride
                w = (w * 2.0)**2 * a_w
                h = (h * 2.0)**2 * a_h

                x0 = x - w / 2
                y0 = y - h / 2

                self.scores_pro.append(float(scores[idx]))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(labels[idx]))

    def __yolox_decode(self,
                       feats: List[ndarray],
                       conf_thres: float,
                       num_labels: int = 80,
                       **kwargs):
        for i, feat in enumerate(feats):
            stride = 8 << i
            score_feat, box_feat, conf_feat = np.split(
                feat, [num_labels, num_labels + 4], -1)
            conf_feat = sigmoid(conf_feat)

            hIdx, wIdx, _ = np.where(conf_feat > conf_thres)

            num_proposal = hIdx.size
            if not num_proposal:
                continue

            score_feat = sigmoid(score_feat[hIdx, wIdx]) * conf_feat[hIdx,
                                                                     wIdx]
            boxes = box_feat[hIdx, wIdx]
            labels = score_feat.argmax(-1)
            scores = score_feat.max(-1)
            indices = np.where(scores > conf_thres)[0]

            if len(indices) == 0:
                continue

            for idx in indices:
                score = scores[idx]
                label = labels[idx]

                x, y, w, h = boxes[idx]

                x = (x + wIdx[idx]) * stride
                y = (y + hIdx[idx]) * stride
                w = np.exp(w) * stride
                h = np.exp(h) * stride

                x0 = x - w / 2
                y0 = y - h / 2

                self.scores_pro.append(float(score))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(label))

    def __ppyoloe_decode(self,
                         feats: List[ndarray],
                         conf_thres: float,
                         num_labels: int = 80,
                         **kwargs):
        reg_max: int = kwargs.get('reg_max', 17)
        dfl = np.arange(0, reg_max, dtype=np.float32)
        for i, feat in enumerate(feats):
            stride = 8 << i
            score_feat, box_feat = np.split(feat, [
                num_labels,
            ], -1)
            score_feat = sigmoid(score_feat)
            _argmax = score_feat.argmax(-1)
            _max = score_feat.max(-1)
            indices = np.where(_max > conf_thres)
            hIdx, wIdx = indices
            num_proposal = hIdx.size
            if not num_proposal:
                continue

            scores = _max[hIdx, wIdx]
            boxes = box_feat[hIdx, wIdx].reshape(num_proposal, 4, reg_max)
            boxes = softmax(boxes, -1) @ dfl
            labels = _argmax[hIdx, wIdx]

            for k in range(num_proposal):
                score = scores[k]
                label = labels[k]

                x0, y0, x1, y1 = boxes[k]

                x0 = (wIdx[k] + 0.5 - x0) * stride
                y0 = (hIdx[k] + 0.5 - y0) * stride
                x1 = (wIdx[k] + 0.5 + x1) * stride
                y1 = (hIdx[k] + 0.5 + y1) * stride

                w = x1 - x0
                h = y1 - y0

                self.scores_pro.append(float(score))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(label))

    def __yolov6_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        for i, feat in enumerate(feats):
            stride = 8 << i
            score_feat, box_feat = np.split(feat, [
                num_labels,
            ], -1)
            score_feat = sigmoid(score_feat)
            _argmax = score_feat.argmax(-1)
            _max = score_feat.max(-1)
            indices = np.where(_max > conf_thres)
            hIdx, wIdx = indices
            num_proposal = hIdx.size
            if not num_proposal:
                continue

            scores = _max[hIdx, wIdx]
            boxes = box_feat[hIdx, wIdx]
            labels = _argmax[hIdx, wIdx]

            for k in range(num_proposal):
                score = scores[k]
                label = labels[k]

                x0, y0, x1, y1 = boxes[k]

                x0 = (wIdx[k] + 0.5 - x0) * stride
                y0 = (hIdx[k] + 0.5 - y0) * stride
                x1 = (wIdx[k] + 0.5 + x1) * stride
                y1 = (hIdx[k] + 0.5 + y1) * stride

                w = x1 - x0
                h = y1 - y0

                self.scores_pro.append(float(score))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(label))

    def __yolov7_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        anchors: Union[List, Tuple] = kwargs.get(
            'anchors',
            [[(12, 16), (19, 36),
              (40, 28)], [(36, 75), (76, 55),
                          (72, 146)], [(142, 110), (192, 243), (459, 401)]])
        self.__yolov5_decode(feats, conf_thres, num_labels, anchors=anchors)

    def __rtmdet_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        for i, feat in enumerate(feats):
            stride = 8 << i
            score_feat, box_feat = np.split(feat, [
                num_labels,
            ], -1)
            score_feat = sigmoid(score_feat)
            _argmax = score_feat.argmax(-1)
            _max = score_feat.max(-1)
            indices = np.where(_max > conf_thres)
            hIdx, wIdx = indices
            num_proposal = hIdx.size
            if not num_proposal:
                continue

            scores = _max[hIdx, wIdx]
            boxes = box_feat[hIdx, wIdx]
            labels = _argmax[hIdx, wIdx]

            for k in range(num_proposal):
                score = scores[k]
                label = labels[k]

                x0, y0, x1, y1 = boxes[k]

                x0 = (wIdx[k] - x0) * stride
                y0 = (hIdx[k] - y0) * stride
                x1 = (wIdx[k] + x1) * stride
                y1 = (hIdx[k] + y1) * stride

                w = x1 - x0
                h = y1 - y0

                self.scores_pro.append(float(score))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(label))

    def __yolov8_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        self.__yolov6_decode(feats, conf_thres, num_labels)
