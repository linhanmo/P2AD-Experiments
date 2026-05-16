import argparse
import os
from typing import Any, Dict, List

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def _unwrap(x):
    if hasattr(x, 'data'):
        return x.data
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
    p.add_argument('--checkpoint', default='/root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth')
    p.add_argument('--work-dir', default='/root/rivermind-data/PoseBH/experiments/QAT/work_dirs/hrnet_w48_qat')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--calib-disable-after-epoch', type=int, default=1)
    p.add_argument('--save-name', default='qat_int8.pth')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    train_dataset = build_dataset(cfg.data.train, dict(test_mode=False))
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.get('samples_per_gpu', 32),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        shuffle=True,
        drop_last=True,
        dist=False,
        seed=cfg.get('seed', None),
    )

    base = build_posenet(cfg.model)
    load_checkpoint(base, args.checkpoint, map_location='cpu')
    wrapper = QATHeatmapModel(base)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    wrapper.to(device)
    wrapper.train()

    torch.backends.cudnn.benchmark = True
    wrapper.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(wrapper, inplace=True)

    opt = torch.optim.Adam(wrapper.parameters(), lr=args.lr)

    it = 0
    for epoch in range(args.epochs):
        if epoch >= args.calib_disable_after_epoch:
            torch.ao.quantization.disable_observer(wrapper)
            torch.ao.quantization.freeze_bn_stats(wrapper)
        for data in train_loader:
            img = _unwrap(data['img'])
            target = _unwrap(data['target'])
            target_weight = _unwrap(data['target_weight'])
            if isinstance(img, list):
                img = img[0]
            if isinstance(target, list):
                target = target[0]
            if isinstance(target_weight, list):
                target_weight = target_weight[0]
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = target_weight.to(device, non_blocking=True)

            out = wrapper(img)
            loss_dict = base.keypoint_head.get_loss(out, target, target_weight)
            loss = sum(v for v in loss_dict.values())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            it += 1
            if args.log_interval > 0 and (it % args.log_interval == 0):
                print(f'epoch={epoch} iter={it} loss={float(loss):.6f}')

    wrapper.eval().to('cpu')
    int8_model = torch.ao.quantization.convert(wrapper, inplace=False)

    out_path = os.path.join(args.work_dir, args.save_name)
    torch.save({'state_dict': int8_model.state_dict()}, out_path)
    print(out_path)


if __name__ == '__main__':
    main()
