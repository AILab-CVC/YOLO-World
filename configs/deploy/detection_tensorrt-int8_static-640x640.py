_base_ = [
    '../../third_party/mmdeploy/configs/mmdet/_base_/base_static.py',
    '../../third_party/mmdeploy/configs/_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=(640, 640))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])

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
