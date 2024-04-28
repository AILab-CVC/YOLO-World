## YOLO-World Demo

### Getting Started

Setting `PYTHONPATH` as the path to `YOLO-World` and run:

```bash
PYTHONPATH=/xxxx/YOLO-World python demo/yyyy_demo.py
# or directly
PYTHONPATH=./ python demo/yyyy_demo.py
```

#### Gradio Demo

We provide the [Gradio](https://www.gradio.app/) demo for local devices:

```bash
pip install gradio==4.16.0
python demo/demo.py path/to/config path/to/weights
```

Additionaly, you can use a Dockerfile to build an image with gradio. As a prerequisite, make sure you have respective drivers installed alongside [nvidia-container-runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime). Replace MODEL_NAME and WEIGHT_NAME with the respective values or ommit this and use default values from the [Dockerfile](Dockerfile#3)

```bash
docker build --build-arg="MODEL=MODEL_NAME" --build-arg="WEIGHT=WEIGHT_NAME" -t yolo_demo .
docker run --runtime nvidia -p 8080:8080
```

#### Image Demo

We provide a simple image demo for inference on images with visualization outputs.

```bash
python demo/image_demo.py path/to/config path/to/weights image/path/directory 'person,dog,cat' --topk 100 --threshold 0.005 --output-dir demo_outputs
```

**Notes:**
* The `image` can be a directory or a single image.
* The `texts` can be a string of categories (noun phrases) which is separated by a comma. We also support `txt` file in which each line contains a category ( noun phrases).
* The `topk` and `threshold` control the number of predictions and the confidence threshold.


#### Video Demo

The `video_demo` has similar hyper-parameters with `image_demo`.

```bash
python demo/video_demo.py path/to/config path/to/weights video_path 'person,dog' --out out_video_path
```

### FAQ

> 1. `Failed to custom import!`
```bash
  File "simple_demo.py", line 37, in <module>
    cfg = Config.fromfile(config_file)
  File "/data/miniconda3/envs/det/lib/python3.8/site-packages/mmengine/config/config.py", line 183, in fromfile
    raise ImportError('Failed to custom import!') from e
ImportError: Failed to custom import!
```
**Solution:**

```bash
PYTHONPATH=/xxxx/YOLO-World python demo/simple_demo.py
```