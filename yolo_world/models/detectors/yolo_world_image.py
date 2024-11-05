# Copyright (c) Tencent Inc. All rights reserved.
import random
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from transformers import CLIPVisionModelWithProjection


class ImagePromptEncoder(nn.Module):

    def __init__(self,
                 vision_encoder="openai/clip-vit-base-patch32",
                 img_size=224,
                 dim=512) -> None:
        super().__init__()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            vision_encoder)
        self.projector = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                       nn.Linear(dim * 2, dim))
        mean = torch.tensor([0.48145466, 0.4578275,
                             0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258,
                            0.27577711]).view(1, 3, 1, 1)
        self.register_buffer('img_mean', mean)
        self.register_buffer('img_std', std)
        self.freeze_encoder()

    def freeze_encoder(self):
        for _, module in self.image_encoder.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward_vision_encoder(self, images):
        self.image_encoder.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            embeddings = self.image_encoder(images).image_embeds
            embeddings = embeddings / embeddings.norm(
                p=2, dim=-1, keepdim=True)
        return embeddings

    def train(self, mode=True):
        super().train(mode)
        self.freeze_encoder()

    def transform(self, image, bboxes_per_image):

        def scale_bbox(bbox, scale, image_width, image_height):
            x1, y1, x2, y2 = bbox
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            sw, sh = width * scale, height * scale
            x1 = max(0, int(cx - sw / 2.0))
            x2 = min(image_width - 1, int(cx + sw / 2.0))
            y1 = max(0, int(cy - sh / 2.0))
            y2 = min(image_height - 1, int(cy + sh / 2.0))
            return (x1, y1, x2, y2)

        height, width = image.shape[-2:]
        region_images = []

        for bbox in bboxes_per_image:
            scale = min(random.random(), 0.7) + 0.8
            scaled_bbox = scale_bbox(bbox,
                                     scale=scale,
                                     image_width=width,
                                     image_height=height)
            left, top, right, bottom = scaled_bbox
            region_image = image[:, top:bottom + 1,
                                 left:right + 1].unsqueeze(0)
            # print(region_image.shape)
            region_images.append(
                F.interpolate(region_image,
                              size=(224, 224),
                              mode='bilinear',
                              align_corners=False))

        region_images = torch.cat(region_images, dim=0)
        # forward images
        region_images = (region_images - self.img_mean) / self.img_std
        # kxd
        image_embeddings = self.forward_vision_encoder(region_images)
        return image_embeddings

    def forward(self, images, bboxes, class_inds, embeddings=None):
        # image: Bx3xHxW
        # bboxes: List[Tensor[Kx4]]
        batch_size = images.shape[0]
        N, D = embeddings.shape[1:]
        embeddings = embeddings.reshape(batch_size, N, D).clone()
        for ind in range(batch_size):
            bboxes_per_img = bboxes[ind]
            if bboxes_per_img is None or len(bboxes_per_img) == 0:
                embeddings = embeddings + 0.0 * self.projector(embeddings)
                continue
            class_inds_per_image = class_inds[ind]
            # print(class_inds)
            sample_embeddings = self.transform(images[ind], bboxes_per_img)
            sample_embeddings = self.projector(
                sample_embeddings) + sample_embeddings
            embeddings[ind, class_inds_per_image] = embeddings[
                ind, class_inds_per_image] * 0 + sample_embeddings
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings


def sample_random_class_bboxes(bbox_labels, batch_size, min_area=64 * 64):
    # 6x4
    img_inds = bbox_labels[:, 0].long()
    cls_inds = bbox_labels[:, 1].long()
    bboxes = bbox_labels[:, 2:]
    sample_bboxes = []
    sample_cls = []
    for i in range(batch_size):
        image_ind = img_inds == i
        if len(image_ind) == 0:
            sample_bboxes.append([])
            sample_cls.append([])
            continue
        bbox_per_image = bboxes[image_ind]
        if len(bbox_per_image) == 0:
            sample_bboxes.append([])
            sample_cls.append([])
            continue
        area_per_image = (bbox_per_image[:, 2:] -
                          bbox_per_image[:, :2]).prod(dim=1)
        cls_per_image = cls_inds[image_ind]
        # print(cls_per_image.shape, cls_per_image)
        unique_cls = torch.unique(cls_per_image)
        sample_bboxes_per_image = []
        sample_cls_per_image = []
        for class_id in unique_cls:
            class_bbox_inds = ((class_id == cls_per_image) &
                               (area_per_image >= min_area)).nonzero()
            if len(class_bbox_inds) > 0:
                class_bbox_inds = class_bbox_inds[:, 0]
            else:
                continue
            sample_bbox_ind = class_bbox_inds[torch.randint(
                0, class_bbox_inds.shape[0], (1, )).item()]
            sample_bboxes_per_image.append(bbox_per_image[sample_bbox_ind])
            sample_cls_per_image.append(class_id)
        if len(sample_bboxes_per_image) > 0:
            sample_cls_per_image = torch.stack(sample_cls_per_image)
            sample_bboxes_per_image = torch.stack(sample_bboxes_per_image)
        else:
            sample_cls_per_image = None
            sample_bboxes_per_image = None
        sample_bboxes.append(sample_bboxes_per_image)
        sample_cls.append(sample_cls_per_image)
    return sample_cls, sample_bboxes


@MODELS.register_module()
class YOLOWorldImageDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 vision_model='openai/clip-vit-base-patch32',
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        super().__init__(*args, **kwargs)
        self.has_embed = False
        self.txt_feats = None
        self.image_prompt_encoder = ImagePromptEncoder(
            vision_encoder=vision_model)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = txt_feats[0].shape[0]

        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def setembeddings(self, embeddings):
        if embeddings is not None:
            self.txt_feats = embeddings
            self.has_embed = True
        else:
            self.has_embed = False

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        if self.has_embed:
            img_feats, _ = self.backbone(batch_inputs, None)
            txt_feats = self.txt_feats
        else:
            if isinstance(batch_data_samples,
                          dict) and 'texts' in batch_data_samples:
                texts = batch_data_samples['texts']
            elif isinstance(batch_data_samples, list) and hasattr(
                    batch_data_samples[0], 'texts'):
                texts = [
                    data_sample.texts for data_sample in batch_data_samples
                ]
            else:
                texts = None
        if texts is not None:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        else:
            if self.training:
                if txt_feats is not None:
                    # forward image only
                    img_feats = self.backbone.forward_image(batch_inputs)
                else:
                    img_feats, txt_feats = self.backbone(batch_inputs, texts)
            else:
                img_feats = self.backbone.forward_image(batch_inputs)
            if self.training:
                sample_cls_inds, sample_bboxes = sample_random_class_bboxes(
                    batch_data_samples['bboxes_labels'], img_feats[0].shape[0])
                txt_feats = self.image_prompt_encoder(
                    batch_inputs, sample_bboxes, sample_cls_inds, txt_feats,
                    batch_data_samples['image_prompts'])
            # projector will be forwarded outside the module
            # else:
            #     txt_feats = self.image_prompt_encoder.forward_projector(
            #         self.test_embeddings)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
