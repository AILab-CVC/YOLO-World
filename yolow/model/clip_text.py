# Copyright (c) Tencent Inc. All rights reserved.
import itertools
import torch.nn as nn
import warnings
from torch import Tensor
from typing import List, Sequence

# To avoid warnings from huggingface transformers (seems a bug)
# FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0.
# Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

__all__ = ('HuggingCLIPLanguageBackbone', )


class HuggingCLIPLanguageBackbone(nn.Module):

    def __init__(self, model_name: str, frozen_modules: Sequence[str] = (), dropout: float = 0.0) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), ('number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
