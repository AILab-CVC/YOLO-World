import os
import json
import argparse
import os.path as osp

import cv2
import tqdm
import torch
import numpy as np
import tensorflow as tf
import supervision as sv
from torchvision.ops import nms

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser('YOLO-World TFLite (INT8) Demo')
    parser.add_argument('path', help='TFLite Model `.tflite`')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'detecting texts (str, txt, or json), should be consistent with the ONNX model'
    )
    parser.add_argument('--output-dir',
                        default='./output',
                        help='directory to save output files')
    args = parser.parse_args()
    return args


def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)


def generate_anchors_per_level(feat_size, stride, offset=0.5):
    h, w = feat_size
    shift_x = (torch.arange(0, w) + offset) * stride
    shift_y = (torch.arange(0, h) + offset) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x)
    anchors = torch.stack([xx, yy]).reshape(2, -1).transpose(0, 1)
    return anchors


def generate_anchors(feat_sizes=[(80, 80), (40, 40), (20, 20)],
                     strides=[8, 16, 32],
                     offset=0.5):
    anchors = [
        generate_anchors_per_level(fs, s, offset)
        for fs, s in zip(feat_sizes, strides)
    ]
    anchors = torch.cat(anchors)
    return anchors


def simple_bbox_decode(points, pred_bboxes, stride):

    pred_bboxes = pred_bboxes * stride[None, :, None]
    x1 = points[..., 0] - pred_bboxes[..., 0]
    y1 = points[..., 1] - pred_bboxes[..., 1]
    x2 = points[..., 0] + pred_bboxes[..., 2]
    y2 = points[..., 1] + pred_bboxes[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return bboxes


def visualize(image, bboxes, labels, scores, texts):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def inference_per_sample(interp,
                         image_path,
                         texts,
                         priors,
                         strides,
                         output_dir,
                         size=(640, 640),
                         vis=False,
                         score_thr=0.05,
                         nms_thr=0.3,
                         max_dets=300):

    # input / output details from TFLite
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # load image from path
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)

    # inference
    interp.set_tensor(input_details[0]['index'], image)
    interp.invoke()

    scores = interp.get_tensor(output_details[1]['index'])
    bboxes = interp.get_tensor(output_details[0]['index'])

    # can be converted to numpy for other devices
    # using torch here is only for references.
    ori_scores = torch.from_numpy(scores[0])
    ori_bboxes = torch.from_numpy(bboxes)

    # decode bbox cordinates with priors
    decoded_bboxes = simple_bbox_decode(priors, ori_bboxes, strides)[0]
    scores_list = []
    labels_list = []
    bboxes_list = []
    for cls_id in range(len(texts)):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(decoded_bboxes, cls_scores, iou_threshold=0.5)
        cur_bboxes = decoded_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    # only for visualization, add an extra NMS
    keep_idxs = nms(bboxes, scores, iou_threshold=nms_thr)
    num_dets = min(len(keep_idxs), max_dets)
    bboxes = bboxes[keep_idxs].unsqueeze(0)
    scores = scores[keep_idxs].unsqueeze(0)
    labels = labels[keep_idxs].unsqueeze(0)

    scores = scores[0, :num_dets].numpy()
    bboxes = bboxes[0, :num_dets].numpy()
    labels = labels[0, :num_dets].numpy()

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)

    if vis:
        image_out = visualize(ori_image, bboxes, labels, scores, texts)
        cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)
        print(f"detecting {num_dets} objects.")
        return image_out, ori_scores, ori_bboxes[0]
    else:
        return bboxes, labels, scores


def main():

    args = parse_args()
    tflite_file = args.tflite
    # init ONNX session
    interpreter = tf.lite.Interpreter(model_path=tflite_file,
                                      experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    print("Init TFLite Interpter")
    output_dir = "onnx_outputs"
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines]
    elif args.text.endswith('.json'):
        texts = json.load(open(args.text))
    else:
        texts = [[t.strip()] for t in args.text.split(',')]

    size = (640, 640)
    strides = [8, 16, 32]

    # prepare anchors, since TFLite models does not contain anchors, due to INT8 quantization.
    featmap_sizes = [(size[0] // s, size[1] // s) for s in strides]
    flatten_priors = generate_anchors(featmap_sizes, strides=strides)
    mlvl_strides = [
        flatten_priors.new_full((featmap_size[0] * featmap_size[1] * 1, ),
                                stride)
        for featmap_size, stride in zip(featmap_sizes, strides)
    ]
    flatten_strides = torch.cat(mlvl_strides)

    print("Start to inference.")
    for img in tqdm.tqdm(images):
        inference_per_sample(interpreter,
                             img,
                             texts,
                             flatten_priors[None],
                             flatten_strides,
                             output_dir=output_dir,
                             vis=True,
                             score_thr=0.3,
                             nms_thr=0.5)
    print("Finish inference")


if __name__ == "__main__":
    main()
