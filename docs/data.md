## Preparing Data for YOLO-World


### Overview

For pre-training YOLO-World, we adopt several datasets as listed in the below table:

| Data | Samples | Type | Boxes  |
| :-- | :-----: | :---:| :---: | 
| Objects365v1 | 609k | detection | 9,621k |
| GQA | 621k | grounding | 3,681k |
| Flickr | 149k | grounding | 641k |
| CC3M-Lite | 245k | image-text | 821k |


### Dataset Class

For training YOLO-World, we mainly adopt two kinds of dataset classs:

#### 1. `MultiModalDataset`

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


#### 2. `MixedGroundingDataset`




