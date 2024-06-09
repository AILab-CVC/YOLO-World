## Prompt YOLO-World


### 1. Simple YOLO-World with Embeddings

For simplifying YOLO-World and get rid of the language model, we define a new basic detector `SimpleYOLOWorldDetector`:

The `SimpleYOLOWorldDetector` supports prompt embeddings as the input and doesn't not contain a language model anymore!
Now, YOLO-World adopts `embeddings` as language inputs, and the embeddings support several kinds: (1) text embeddings from the language model, e.g., CLIP language encoder, (2) image embeddings from a vision model, e.g., CLIP vision encoder, and (3) image-text fused embeddings, and (4) random embeddings.
The (1)(2)(3) supports zero-shot inference and (4), including (1)(2)(3) are designed for prompt tuning on your custom data.

The basic detector is defined as follows:

```python
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs)
```

To use it in a zero-shot manner, you need to pre-compute the text embeddings (image embeddings) and save it as a `numpy array (*.npy)` with a `NxD` shape (N is the number of prompts and D is the dimension of the embeddings). Currently, we only support one prompt for one class. You can use several prompts for one class but you need to merge the results in the post-processing steps.


### 2. Prompt Tuning YOLO-World

We introduce prompt tuning for YOLO-World to maintain the zero-shot ability while improve the performance on your custom datasets.

For more details about writing configs for prompt tuning, you can refer to [`prompt tuning for COCO data`](./../configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py).

1. Use random prompts

```python
dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             prompt_dim=text_channels,
             num_prompts=80,
             ...)
```

2. Use CLIP embeddings (text, image, or text-image embeddings)

the `clip_vit_b32_coco_80_embeddings.npy` can be downloaded at [HuggingFace](https://huggingface.co/wondervictor/YOLO-World/blob/main/clip_vit_b32_coco_80_embeddings.npy).

```python
dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path='embeddings/clip_vit_b32_coco_80_embeddings.npy',
             prompt_dim=text_channels,
             num_prompts=80,
             ...)
```

Using CLIP model to obtains the image and text embeddings will maintain the zero-shot performace.


| Model | Config |  AP  | AP50 | AP75  | APS | APM | APL |
| :---- | :----: | :--: | :--: | :---: | :-: | :-: | :-: |
| YOLO-World-v2-L | Zero-shot | 45.7 | 61.6 | 49.8 | 29.9 | 50.0 | 60.8 |
| [YOLO-World-v2-L](./../configs/prompt_tuning_coco/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_mask-refine_prompt_tuning_coco.py) | Prompt tuning | 47.9 | 64.3 | 52.5 | 31.9 | 52.6 | 61.3 | 
