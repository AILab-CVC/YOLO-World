## Run YOLO-World (Quantized) on TF-Lite

- [x] Export YOLO-World to TFLite with INT8 Quantization.
- [x] TFLite demo

### Priliminaries

```bash
pip install onnxruntime onnx onnx-simplifier
pip install tensorflow==2.15.1
```

See [onnx2tf](https://github.com/PINTO0309/onnx2tf) for more details about export TFLite models.
The contributor of `onnx2tf` is very nice!

### Export TFLite INT8 Quantization models 

Please use **Reparameterized YOLO-World** for TFLite!!

1. Prepare the ONNX model

Please export the ONNX model without `postprocessing` and `bbox_decoder`, just add `--without-bbox-decoder`!
`bbox_decoder` is not supported for INT8 quantization, please take care!

```bash
PYTHONPATH=./ python deploy/export_onnx.py path/to/config path/to/weights --custom-text path/to/customtexts --opset 11 --without-bbox-decoder
```

2. Generate the calibration samples

Using 100 COCO images is suggested to create a simple calibration dataset for quantization.

```python
import os
import random
from PIL import Image, ImageOps
import cv2
import glob
import numpy as np

root = "data/coco/val2017/"
image_list = os.listdir(root)
image_list = [os.path.join(root, f) for f in image_list]
random.shuffle(image_list)

img_datas = []
for idx, file in enumerate(image_list[:100]):
    image = Image.open(file).convert('RGB')
    # Get sample input data as a numpy array in a method of your choosing.
    img_width, img_height = image.size
    size = max(img_width, img_height)
    image = ImageOps.pad(image, (size, size), method=Image.BILINEAR)
    image = image.resize((640, 640), Image.BILINEAR)
    tensor_image = np.asarray(image).astype(np.float32)
    tensor_image /= 255.0
    tensor_image = np.expand_dims(tensor_image, axis=0)
    img_datas.append(tensor_image)

calib_datas = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_datas.shape}')
np.save(file='tflite_calibration_data_100_images_640.npy', arr=calib_datas)

```

3. Export ONNX to TFLite using `onnx2tf`

```bash
onnx2tf -i [ONNX] -o [OUTPUT] -oiqt  -cind "images" "tflite_calibration_data_100_images_640.npy" "[[[[0.,0.,0.]]]]" "[[[[1.,1.,1.]]]]"  -onimc "scores" "bboxes" --verbosity debug
```

We provide a sample TFLite INT8 model: [yolo_world_x_coco_zeroshot_rep_integer_quant.tflite](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_x_coco_zeroshot_rep_integer_quant.tflite)

### Inference using TFLite

```bash
python deploy/tflite_demo.py path/to/tflite path/to/images path/to/texts

```