## YOLO-World (w/o third-party dependencies)

### Install `yolow`

```bash
pip install -r requirements.txt
pip install git+https://github.com/lvis-dataset/lvis-api.git
python setup.py develop
```

### Preparation

- Prepare datasets under `./data` folder following [docs/data.md](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/data.md)
- Download pretrained checkpoints under `./pretrained_weights` folder, e.g.,
```bash
wget -P pretrained_weights/ https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth
```
- Check the installation via unit test, i.e.,
```bash
python scripts/unit_test.py
```

### Test Scripts

1. Test pretrained YOLO-World models on LVIS dataset:
```bash
# ./scripts/dist_test.sh {model_size} {checkpoint_path} {gpu_num}
./scripts/dist_test.sh s ./pretrained_weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth 8
```

2. Test single image:
```bash
python scripts/inference.py
```
