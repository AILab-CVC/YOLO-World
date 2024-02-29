_base_ = (
    '../../third_party/mmdeploy/configs/mmdet/detection/'
    'detection_onnxruntime-fp16_dynamic.py')
backend_config = dict(
    precision='int8')
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.1,
        confidence_threshold=0.005,
        iou_threshold=0.3,
        max_output_boxes_per_class=100,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
backend_config = dict(
    type='onnxruntime')
