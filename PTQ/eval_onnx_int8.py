import argparse
from typing import Any, Dict, List

import numpy as np
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default='/root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth')
    p.add_argument('--onnx', required=True)
    p.add_argument('--eval', nargs='+', default=['mAP'])
    p.add_argument('--flip-test', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
        drop_last=False,
        dist=False,
        seed=cfg.get('seed', None),
    )
    data_loader = build_dataloader(dataset, **loader_cfg)

    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    results: List[Dict[str, Any]] = []
    for data in data_loader:
        img = _unwrap(data['img'])
        if isinstance(img, list):
            img = img[0]
        img_np = img.detach().cpu().numpy().astype(np.float32)

        img_metas = _unwrap_img_metas(data['img_metas'])
        img_h = int(img_np.shape[2])
        img_w = int(img_np.shape[3])

        heatmap = sess.run(None, {input_name: img_np})[0]

        flip_test = args.flip_test or cfg.model.get('test_cfg', {}).get('flip_test', False)
        if flip_test:
            img_flip = img_np[:, :, :, ::-1].copy()
            heatmap_flip = sess.run(None, {input_name: img_flip})[0]
            flip_pairs = img_metas[0].get('flip_pairs', [])
            target_type = cfg.model.get('test_cfg', {}).get('target_type', 'GaussianHeatmap')
            heatmap_flip = flip_back(heatmap_flip, flip_pairs, target_type=target_type)
            heatmap = (heatmap + heatmap_flip) * 0.5

        out = model.keypoint_head.decode(img_metas, heatmap, img_size=[img_w, img_h])
        results.append(out)

    eval_config = cfg.get('evaluation', {})
    metric = args.eval
    eval_kwargs = {}
    if isinstance(eval_config, dict):
        eval_kwargs.update(eval_config)
    eval_kwargs['metric'] = metric

    res = dataset.evaluate(results, '.', **eval_kwargs)
    for k, v in sorted(res.items()):
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()

