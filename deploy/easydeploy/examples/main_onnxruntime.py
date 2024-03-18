import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import onnxruntime
from config import (CLASS_COLORS, CLASS_NAMES, ModelType, YOLOv5_ANCHORS,
                    YOLOv7_ANCHORS)
from cv2_nms import non_max_suppression
from numpy_coder import Decoder
from preprocess import Preprocess
from tqdm import tqdm

# Add __FILE__  to sys.path
sys.path.append(str(Path(__file__).resolve().parents[0]))

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def path_to_list(path: str):
    path = Path(path)
    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        res_list = [str(path.absolute())]
    elif path.is_dir():
        res_list = [
            str(p.absolute()) for p in path.iterdir()
            if p.suffix in IMG_EXTENSIONS
        ]
    else:
        raise RuntimeError
    return res_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('onnx', type=str, help='Onnx file')
    parser.add_argument('--type', type=str, help='Model type')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument(
        '--out-dir', default='./output', type=str, help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.7, help='Bbox iou threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_type = ModelType(args.type.lower())

    if not args.show:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = path_to_list(args.img)
    session = onnxruntime.InferenceSession(
        args.onnx, providers=['CPUExecutionProvider'])
    preprocessor = Preprocess(model_type)
    decoder = Decoder(model_type, model_only=True)
    if model_type == ModelType.YOLOV5:
        anchors = YOLOv5_ANCHORS
    elif model_type == ModelType.YOLOV7:
        anchors = YOLOv7_ANCHORS
    else:
        anchors = None

    for file in tqdm(files):
        image = cv2.imread(file)
        image_h, image_w = image.shape[:2]
        img, (ratio_w, ratio_h) = preprocessor(image, args.img_size)
        features = session.run(None, {'images': img})
        decoder_outputs = decoder(
            features,
            args.score_thr,
            num_labels=len(CLASS_NAMES),
            anchors=anchors)
        nmsd_boxes, nmsd_scores, nmsd_labels = non_max_suppression(
            *decoder_outputs, args.score_thr, args.iou_thr)
        for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
            x0, y0, x1, y1 = box
            x0 = math.floor(min(max(x0 / ratio_w, 1), image_w - 1))
            y0 = math.floor(min(max(y0 / ratio_h, 1), image_h - 1))
            x1 = math.ceil(min(max(x1 / ratio_w, 1), image_w - 1))
            y1 = math.ceil(min(max(y1 / ratio_h, 1), image_h - 1))
            cv2.rectangle(image, (x0, y0), (x1, y1), CLASS_COLORS[label], 2)
            cv2.putText(image, f'{CLASS_NAMES[label]}: {score:.2f}',
                        (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        if args.show:
            cv2.imshow('result', image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(f'{out_dir / Path(file).name}', image)


if __name__ == '__main__':
    main()
