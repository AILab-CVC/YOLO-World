## Update Notes

We provide the details for important updates of YOLO-World in this note.

### Model Architecture

**[2024-2-29]:** YOLO-World-v2:

1. We remove the `I-PoolingAttention`: though it improves the performance for zero-shot LVIS evaluation, it affects the inference speeds after exporting YOLO-World to ONNX or TensorRT. Considering the trade-off, we remove the `I-PoolingAttention` in the newest version.
2. We replace the `L2-Norm` in the contrastive head with the `BatchNorm`. The `L2-Norm` contains complex operations, such as `reduce`, which is time-consuming for deployment. However, the `BatchNorm` can be fused into the convolution, which is much more efficient and also improves the zero-shot performance.




