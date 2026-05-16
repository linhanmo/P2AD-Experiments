import argparse
import os

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmpose.models import build_posenet


class HeatmapOnlyModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = base.backbone
        self.neck = base.neck if hasattr(base, 'neck') else None
        self.keypoint_head = base.keypoint_head

    def forward(self, img):
        feats = self.backbone(img)
        if self.neck is not None:
            feats = self.neck(feats)
        out = self.keypoint_head.inference_model(feats, flip_pairs=None)
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default='/root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    wrapper = HeatmapOnlyModel(model).eval()
    if args.device == 'cuda':
        wrapper = wrapper.cuda()

    img_h, img_w = 256, 192
    dummy = torch.randn(1, 3, img_h, img_w, device=next(wrapper.parameters()).device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=['img'],
        output_names=['heatmap'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={'img': {0: 'batch'}, 'heatmap': {0: 'batch'}},
    )
    print(args.out)


if __name__ == '__main__':
    main()

