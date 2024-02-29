_base_ = (
    '../../third_party/mmyolo/configs/deploy/'
    'detection_onnxruntime_static.py')
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.25,
        confidence_threshold=0.005,
        iou_threshold=0.65,
        max_output_boxes_per_class=200,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
backend_config = dict(
    type='onnxruntime')
