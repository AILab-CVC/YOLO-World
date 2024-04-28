import os
import json
import argparse
import os.path as osp

import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort
from mmengine.utils import ProgressBar

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()


def parse_args():
    parser = argparse.ArgumentParser('YOLO-World ONNX Demo')
    parser.add_argument('onnx', help='onnx file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'detecting texts (str, txt, or json), should be consistent with the ONNX model'
    )
    parser.add_argument('--output-dir',
                        default='./output',
                        help='directory to save output files')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference')
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


def visualize(image, bboxes, labels, scores, texts):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def inference(ort_session, image_path, texts, output_dir, size=(640, 640)):
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["num_dets", "labels", "scores", "boxes"],
                              {"images": input_ort})
    num_dets, labels, scores, bboxes = results
    num_dets = num_dets[0][0]
    labels = labels[0, :num_dets]
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, w)
    bboxes = bboxes.round().astype('int')

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)
    return image_out


def main():

    args = parse_args()
    onnx_file = args.onnx
    # init ONNX session
    ort_session = ort.InferenceSession(
        onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Init ONNX Runtime session")
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

    print("Start to inference.")
    progress_bar = ProgressBar(len(images))
    for img in images:
        inference(ort_session, img, texts, output_dir=output_dir)
        progress_bar.update()
    print("Finish inference")


if __name__ == "__main__":
    main()
