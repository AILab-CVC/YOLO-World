from typing import List, Tuple, Union

import cv2
from numpy import ndarray

MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4


def non_max_suppression(boxes: Union[List[ndarray], Tuple[ndarray]],
                        scores: Union[List[float], Tuple[float]],
                        labels: Union[List[int], Tuple[int]],
                        conf_thres: float = 0.25,
                        iou_thres: float = 0.65) -> Tuple[List, List, List]:
    if MINOR >= 7:
        indices = cv2.dnn.NMSBoxesBatched(boxes, scores, labels, conf_thres,
                                          iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres,
                                   iou_thres).flatten()

    nmsd_boxes = []
    nmsd_scores = []
    nmsd_labels = []
    for idx in indices:
        box = boxes[idx]
        # x0y0wh -> x0y0x1y1
        box[2:] = box[:2] + box[2:]
        score = scores[idx]
        label = labels[idx]
        nmsd_boxes.append(box)
        nmsd_scores.append(score)
        nmsd_labels.append(label)
    return nmsd_boxes, nmsd_scores, nmsd_labels
