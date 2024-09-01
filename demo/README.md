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

Additionally, you can use our Docker build system for an easier setup:

```bash
./build_and_run.sh <model-key>
```

Available model keys include:
- seg-l, seg-l-seghead, seg-m, seg-m-seghead
- pretrain-l-clip-800ft, pretrain-l-clip, pretrain-l-1280ft, pretrain-l
- pretrain-m-1280ft, pretrain-m, pretrain-s-1280ft, pretrain-s
- pretrain-x-cc3mlite, pretrain-x-1280ft

This script will build the Docker image and run the container with the specified model configuration. The Gradio interface will be accessible at `http://localhost:8080`.

You can also customize the model weights directory by setting the `MODEL_DIR` environment variable:

```bash
MODEL_DIR=/path/to/your/weights ./build_and_run.sh <model-key>
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