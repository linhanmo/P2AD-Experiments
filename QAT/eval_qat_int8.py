import argparse
from typing import Any, Dict, List

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmpose.core.post_processing import flip_back
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def _unwrap(x):
    if hasattr(x, 'data'):
        return x.data
    return x


def _unwrap_img_metas(x):
    x = _unwrap(x)
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    return x


class QATHeatmapModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.backbone = base.backbone
        self.neck = base.neck if hasattr(base, 'neck') else None
        self.keypoint_head = base.keypoint_head
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, img):
        img = self.quant(img)
        feats = self.backbone(img)
        if self.neck is not None:
            feats = self.neck(feats)
        out = self.keypoint_head(feats)
        out = self.dequant(out)
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--float-checkpoint', default='/root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth')
    p.add_argument('--qat-int8', required=True)
    p.add_argument('--eval', nargs='+', default=['mAP'])
    p.add_argument('--flip-test', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
        drop_last=False,
        dist=False,
        seed=cfg.get('seed', None),
    )

    base = build_posenet(cfg.model)
    load_checkpoint(base, args.float_checkpoint, map_location='cpu')
    wrapper = QATHeatmapModel(base).eval()
    wrapper.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(wrapper, inplace=True)
    wrapper.eval()
    int8_model = torch.ao.quantization.convert(wrapper, inplace=False).eval()

    sd = torch.load(args.qat_int8, map_location='cpu')
    state = sd.get('state_dict', sd)
    int8_model.load_state_dict(state, strict=True)

    results: List[Dict[str, Any]] = []
    flip_test = args.flip_test or cfg.model.get('test_cfg', {}).get('flip_test', False)
    target_type = cfg.model.get('test_cfg', {}).get('target_type', 'GaussianHeatmap')
    for data in data_loader:
        img = _unwrap(data['img'])
        if isinstance(img, list):
            img = img[0]
        img = img.cpu()
        img_np = img.numpy().astype(np.float32)

        img_metas = _unwrap_img_metas(data['img_metas'])
        img_h = int(img_np.shape[2])
        img_w = int(img_np.shape[3])

        heatmap = int8_model(img).detach().cpu().numpy()
        if flip_test:
            img_flip = img_np[:, :, :, ::-1].copy()
            img_flip_t = torch.from_numpy(img_flip)
            heatmap_flip = int8_model(img_flip_t).detach().cpu().numpy()
            flip_pairs = img_metas[0].get('flip_pairs', [])
            heatmap_flip = flip_back(heatmap_flip, flip_pairs, target_type=target_type)
            heatmap = (heatmap + heatmap_flip) * 0.5

        out = base.keypoint_head.decode(img_metas, heatmap, img_size=[img_w, img_h])
        results.append(out)

    eval_config = cfg.get('evaluation', {})
    eval_kwargs = {}
    if isinstance(eval_config, dict):
        eval_kwargs.update(eval_config)
    eval_kwargs['metric'] = args.eval
    res = dataset.evaluate(results, '.', **eval_kwargs)
    for k, v in sorted(res.items()):
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()

