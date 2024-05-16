## Deploy YOLO-World

- [x] ONNX export
- [x] ONNX demo
- [ ] TensorRT
- [ ] TFLite

We provide several ways to deploy YOLO-World with ONNX or TensorRT

### Priliminaries

```bash
pip install supervision onnx onnxruntime onnxsim
```

### Export ONNX on Gradio Demo

start the `demo.py` and you can modify the texts in the demo and output the ONNX model.

```bash
python demo.py path/to/config path/to/weights
```

### Export YOLO-World to ONNX models

You can also use [`export_onnx.py`](../deploy/export_onnx.py) to obtain the ONNX model. You might specify the `--custom-text` with your own `Text JSON` for your custom prompts. The format of `Text JSON` can be found in [`docs/data`](../docs/data.md).

```bash
PYTHONPATH=./ python deploy/export_onnx.py path/to/config path/to/weights --custom-text path/to/customtexts --opset 11
```

If you don't want to include `NMS` or "post-processing" into the ONNX model, you can add `--without-nms`
```bash
PYTHONPATH=./ python deploy/export_onnx.py path/to/config path/to/weights --custom-text path/to/customtexts --opset 11 --without-nms
```

If you want to quantize YOLO-World with ONNX model, you'd better remove `NMS` and `bbox_decoder` by adding `--without-bbox-decoder`

```bash
PYTHONPATH=./ python deploy/export_onnx.py path/to/config path/to/weights --custom-text path/to/customtexts --opset 11 --without-bbox-decoder
```

**Running ONNX demo**

```bash
python deploy/onnx_demo.py path/to/model.onnx path/to/images path/to/texts
```


### Export YOLO-World to TensorRT models

coming soon.

### FAQ

**Q1**. `RuntimeError: Exporting the operator einsum to ONNX opset version 11 is not supported. Support for this operator was added in version 12, try exporting with this version.`

**A:** This error arises because YOLO-World adopts `einsum` for matrix multiplication while it is not supported by `opset 11`. You can set the `--opset` from `11` to `12` if your device supports or change the `einsum` to normal `permute/reshape/multiplication` by set `use_einsum=False` in the `MaxSigmoidCSPLayerWithTwoConv` and `YOLOWorldHeadModule`. You can refer to the [sample config](../configs/pretrain/yolo_world_v2_m_vlpan_bn_noeinsum_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) without einsum.

