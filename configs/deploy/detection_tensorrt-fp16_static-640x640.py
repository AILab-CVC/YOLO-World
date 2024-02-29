_base_ = (
    '../../third_party/mmyolo/configs/deploy/'
    'detection_tensorrt-fp16_static-640x640.py')
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=(640, 640),
    optimize=True)
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 34),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.25,
        confidence_threshold=0.005,
        iou_threshold=0.65,
        max_output_boxes_per_class=100,
        pre_top_k=1,
        keep_top_k=1,
        background_label_id=-1),
    module=['mmyolo.deploy'])
