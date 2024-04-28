## Preparing Data for YOLO-World

### Overview

For pre-training YOLO-World, we adopt several datasets as listed in the below table:

| Data | Samples | Type | Boxes  |
| :-- | :-----: | :---:| :---: | 
| Objects365v1 | 609k | detection | 9,621k |
| GQA | 621k | grounding | 3,681k |
| Flickr | 149k | grounding | 641k |
| CC3M-Lite | 245k | image-text | 821k |
 
### Dataset Directory

We put all data into the `data` directory, such as:

```bash
├── coco
│   ├── annotations
│   ├── lvis
│   ├── train2017
│   ├── val2017
├── flickr
│   ├── annotations
│   └── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── objects365v1
│   ├── annotations
│   ├── train
│   ├── val
```
**NOTE**: We strongly suggest that you check the directories or paths in the dataset part of the config file, especially for the values `ann_file`, `data_root`, and `data_prefix`.

We provide the annotations of the pre-training data in the below table:

| Data | images | Annotation File |
| :--- | :------| :-------------- |
| Objects365v1 | [`Objects365 train`](https://opendatalab.com/OpenDataLab/Objects365_v1) | [`objects365_train.json`](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [`GQA`](https://nlp.stanford.edu/data/gqa/images.zip) | [`final_mixed_train_no_coco.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [`Flickr30k`](https://shannon.cs.illinois.edu/DenotationGraph/) |[`final_flickr_separateGT_train.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [`COCO val2017`](https://cocodataset.org/) | [`lvis_v1_minival_inserted_image_name.json`](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |

**Acknowledgement:** We sincerely thank [GLIP](https://github.com/microsoft/GLIP) and [mdetr](https://github.com/ashkamath/mdetr) for providing the annotation files for pre-training.


### Dataset Class

> For fine-tuning YOLO-World on Close-set Object Detection, using `MultiModalDataset` is recommended.

#### Setting CLASSES/Categories

If you use `COCO-format` custom datasets, you "DO NOT" need to define a dataset class for custom vocabularies/categories.
Explicitly setting the CLASSES in the config file through `metainfo=dict(classes=your_classes),` is simple:

```python

coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=your_classes),
        data_root='data/your_data',
        ann_file='annotations/your_annotation.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/your_class_texts.json',
    pipeline=train_pipeline)
```


For training YOLO-World, we mainly adopt two kinds of dataset classs:

#### 1. `MultiModalDataset`

`MultiModalDataset` is a simple wrapper for pre-defined Dataset Class, such as `Objects365` or `COCO`, which add the texts (category texts) into the dataset instance for formatting input texts.  

**Text JSON**

The json file is formatted as follows:

```json
[
    ['A_1','A_2'],
    ['B'],
    ['C_1', 'C_2', 'C_3'],
    ...
]
```

We have provided the text json for [`LVIS`](./../data/texts/lvis_v1_class_texts.json), [`COCO`](../data/texts/coco_class_texts.json), and [`Objects365`](../data/texts/obj365v1_class_texts.json)

#### 2. `YOLOv5MixedGroundingDataset`

The `YOLOv5MixedGroundingDataset` extends the `COCO` dataset by supporting loading texts/captions from the json file. It's desgined for `MixedGrounding` or `Flickr30K` with text tokens for each object.



### 🔥 Custom Datasets

For custom dataset, we suggest the users convert the annotation files according to the usage. Note that, converting the annotations to the **standard COCO format** is basically required.

1. **Large vocabulary, grounding, referring:** you can follow the annotation format as the `MixedGrounding` dataset, which adds `caption` and `tokens_positive` for assigning the text for each object. The texts can be a category or a noun phrases.

2. **Custom vocabulary (fixed):** you can adopt the `MultiModalDataset` wrapper as the `Objects365` and create a **text json** for your custom categories.


### CC3M Pseudo Annotations

The following annotations are generated according to the automatic labeling process in our paper. Adn we report the results based on these annotations.

To use CC3M annotations, you need to prepare the `CC3M` images first.

| Data | Images | Boxes | File |
| :--: | :----: | :---: | :---: |
| CC3M-246K | 246,363 | 820,629 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_annotations.json) |
| CC3M-500K | 536,405 | 1,784,405| [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_500k_annotations.json) |
| CC3M-750K | 750,000 | 4,504,805 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_750k_annotations.json) |