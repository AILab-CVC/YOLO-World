import os
import argparse

import torch
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser("Reparameterize YOLO-World")
    parser.add_argument('--model', help='model checkpoints to reparameterize')
    parser.add_argument('--out-dir', help='output checkpoints')
    parser.add_argument(
        '--text-embed',
        help='text embeddings to reparameterized into YOLO-World')
    parser.add_argument('--conv-neck',
                        action='store_true',
                        help='whether using 1x1 conv in RepVL-PAN')

    args = parser.parse_args()
    return args


def convert_head(scale, bias, text_embed):
    N, D = text_embed.shape
    weight = (text_embed * scale.exp()).view(N, D, 1, 1)
    bias = torch.ones(N) * bias
    return weight, bias


def reparameterize_head(state_dict, embeds):

    cls_layers = [
        'bbox_head.head_module.cls_contrasts.0',
        'bbox_head.head_module.cls_contrasts.1',
        'bbox_head.head_module.cls_contrasts.2'
    ]

    for i in range(3):
        scale = state_dict[cls_layers[i] + '.logit_scale']
        bias = state_dict[cls_layers[i] + '.bias']
        weight, bias = convert_head(scale, bias, embeds)
        state_dict[cls_layers[i] + '.conv.weight'] = weight
        state_dict[cls_layers[i] + '.conv.bias'] = bias
        del state_dict[cls_layers[i] + '.bias']
        del state_dict[cls_layers[i] + '.logit_scale']
    return state_dict


def convert_neck_split_conv(input_state_dict, block_name, text_embeds,
                            num_heads):
    if block_name + '.guide_fc.weight' not in input_state_dict:
        return input_state_dict
    guide_fc_weight = input_state_dict[block_name + '.guide_fc.weight']
    guide_fc_bias = input_state_dict[block_name + '.guide_fc.bias']
    guide = text_embeds @ guide_fc_weight.transpose(0,
                                                    1) + guide_fc_bias[None, :]
    N, D = guide.shape
    guide = list(guide.split(D // num_heads, dim=1))
    del input_state_dict[block_name + '.guide_fc.weight']
    del input_state_dict[block_name + '.guide_fc.bias']
    for i in range(num_heads):
        input_state_dict[block_name +
                         f'.guide_convs.{i}.weight'] = guide[i][:, :, None,
                                                                None]
    return input_state_dict


def convert_neck_weight(input_state_dict, block_name, embeds, num_heads):
    guide_fc_weight = input_state_dict[block_name + '.guide_fc.weight']
    guide_fc_bias = input_state_dict[block_name + '.guide_fc.bias']
    guide = embeds @ guide_fc_weight.transpose(0, 1) + guide_fc_bias[None, :]
    N, D = guide.shape
    del input_state_dict[block_name + '.guide_fc.weight']
    del input_state_dict[block_name + '.guide_fc.bias']
    input_state_dict[block_name + '.guide_weight'] = guide.view(
        N, D // num_heads, num_heads)
    return input_state_dict


def reparameterize_neck(state_dict, embeds, type='conv'):

    neck_blocks = [
        'neck.top_down_layers.0.attn_block',
        'neck.top_down_layers.1.attn_block',
        'neck.bottom_up_layers.0.attn_block',
        'neck.bottom_up_layers.1.attn_block'
    ]
    if "neck.top_down_layers.0.attn_block.bias" not in state_dict:
        return state_dict
    for block in neck_blocks:
        num_heads = state_dict[block + '.bias'].shape[0]
        if type == 'conv':
            convert_neck_split_conv(state_dict, block, embeds, num_heads)
        else:
            convert_neck_weight(state_dict, block, embeds, num_heads)
    return state_dict


def main():

    args = parse_args()

    # load checkpoint
    model = torch.load(args.model, map_location='cpu')
    state_dict = model['state_dict']

    # load embeddings
    embeddings = torch.from_numpy(np.load(args.text_embed))

    # remove text encoder
    keys = list(state_dict.keys())
    keys = [x for x in keys if "text_model" not in x]

    state_dict_wo_text = {x: state_dict[x] for x in keys}
    print("removing text encoder")

    state_dict_wo_text = reparameterize_head(state_dict_wo_text, embeddings)
    print("reparameterizing head")

    if args.conv_neck:
        neck_type = "conv"
    else:
        neck_type = "linear"

    state_dict_wo_text = reparameterize_neck(state_dict_wo_text, embeddings,
                                             neck_type)

    print("reparameterizing neck")

    model['state_dict'] = state_dict_wo_text

    model_name = os.path.basename(args.model)
    model_name = model_name.replace('.pth', f'_rep_{neck_type}.pth')
    torch.save(model, os.path.join(args.out_dir, model_name))


if __name__ == "__main__":
    main()
