# Copyright (c) Tencent. All rights reserved.
import cv2
import supervision as sv
import torch
from torchvision.ops import nms
from torchvision.transforms import Compose

from yolow.data.transforms import (YOLOResize, LoadAnnotations, LoadImageFromFile, LoadText,
                                   PackDetInputs)
from yolow.model import (build_yolov8_backbone, build_yoloworld_backbone, build_yoloworld_data_preprocessor,
                         build_yoloworld_detector, build_yoloworld_head, build_yoloworld_neck, build_yoloworld_text)

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
mask_annotator = sv.MaskAnnotator()


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


label_annotator = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)

class_names = ("person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
               "traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
               "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
               "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
               "sports ball, kite, baseball bat, baseball glove, skateboard, "
               "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
               "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
               "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, "
               "dining table, toilet, tv, laptop, mouse, remote, keyboard, "
               "cell phone, microwave, oven, toaster, sink, refrigerator, book, "
               "clock, vase, scissors, teddy bear, hair drier, toothbrush")


def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0])
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"


def create_model(model_size, model_file):
    # We have predefined settings in `model/model_cfgs`,
    # including the default architectures for different
    # sizes of YOLO-World models.
    # You can further specify some items via model_args.
    model_args = dict(
        yoloworld_data_preprocessor=dict(),
        yolov8_backbone=dict(),
        yoloworld_text=dict(),
        yoloworld_backbone=dict(),
        yoloworld_neck=dict(),
        yoloworld_head_module=dict(),
        yoloworld_detector=dict(),
    )

    # build model
    data_preprocessor = build_yoloworld_data_preprocessor(model_size, args=model_args)
    yolov8_backbone = build_yolov8_backbone(model_size, args=model_args)
    text_backbone = build_yoloworld_text(model_size, args=model_args)
    yolow_backbone = build_yoloworld_backbone(model_size, yolov8_backbone, text_backbone, args=model_args)
    yolow_neck = build_yoloworld_neck(model_size, args=model_args)
    yolow_head = build_yoloworld_head(model_size, args=model_args)
    yoloworld_model = build_yoloworld_detector(
        model_size, yolow_backbone, yolow_neck, yolow_head, data_preprocessor, args=model_args)

    # load ckpt (mandatory)
    ckpt = torch.load(model_file, map_location="cpu")
    yoloworld_model.load_state_dict(ckpt['state_dict'], strict=True)
    return yoloworld_model


def run_image(
        model,
        input_image,
        max_num_boxes=100,
        score_thr=0.1,
        nms_thr=0.7,
        img_scale=(1280, 1280),
        output_image="./demo_imgs/output.png",
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    pipeline = Compose([
        LoadImageFromFile(),
        YOLOResize(scale=img_scale),
        LoadAnnotations(with_bbox=True),
        LoadText(),
        PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', \
                            'img_shape', 'scale_factor', 'pad_param', 'texts'))
    ])

    data_info = pipeline(dict(img_id=0, img_path=input_image, texts=texts))

    data_batch = dict(
        inputs=[data_info["inputs"]],
        data_samples=[data_info["data_samples"]],
    )

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
        model.class_names = texts
        pred_instances = output['pred_instances']

    # nms
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    new_pred_instances = {
        'scores': [pred_instances.scores[idx] for idx in keep_idxs if pred_instances.scores[idx].float() > score_thr],
        'labels': [pred_instances.labels[idx] for idx in keep_idxs if pred_instances.scores[idx].float() > score_thr],
        'bboxes': [pred_instances.bboxes[idx] for idx in keep_idxs if pred_instances.scores[idx].float() > score_thr]
    }
    new_pred_instances = {key: torch.stack(value).cpu().numpy() for key, value in new_pred_instances.items()}
    pred_instances = new_pred_instances

    if len(pred_instances['scores']) > max_num_boxes:
        indices = pred_instances['scores'].float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output['pred_instances'] = pred_instances

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(
        xyxy=pred_instances['bboxes'], class_id=pred_instances['labels'], confidence=pred_instances['scores'])

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(input_image)
    image = bounding_box_annotator.annotate(image, detections)
    image = label_annotator.annotate(image, detections, labels=labels)
    if masks is not None:
        image = mask_annotator.annotate(image, detections)
    cv2.imwrite(output_image, image)
    print(f"Results saved to {colorstr('bold', output_image)}")


def main():
    # create model
    model = create_model(
        "l",  # [s/m/l/x/xl]
        "./pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth")
    model = model.to('cuda')
    model.eval()

    # start inference
    run_image(model, "./demo_imgs/dog.jpeg")


if __name__ == '__main__':
    main()
