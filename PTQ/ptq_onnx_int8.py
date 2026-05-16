import argparse
import os
from typing import Dict, Iterator, Optional

import numpy as np
from mmcv import Config

from mmpose.datasets import build_dataloader, build_dataset


def _unwrap(x):
    if hasattr(x, 'data'):
        return x.data
    return x


class _CalibReader:
    def __init__(self, data_loader, input_name: str, max_batches: int):
        self.data_loader = data_loader
        self.input_name = input_name
        self.max_batches = max_batches
        self._it = iter(data_loader)
        self._i = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._i >= self.max_batches:
            return None
        try:
            data = next(self._it)
        except StopIteration:
            return None
        img = _unwrap(data['img'])
        if isinstance(img, list):
            img = img[0]
        if hasattr(img, 'detach'):
            img = img.detach()
        img = img.cpu().numpy().astype(np.float32)
        self._i += 1
        return {self.input_name: img}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--onnx-fp32', required=True)
    p.add_argument('--onnx-int8', required=True)
    p.add_argument('--calib-split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--calib-batches', type=int, default=200)
    p.add_argument('--per-channel', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    split_cfg = cfg.data.get(args.calib_split)
    if split_cfg is None:
        raise ValueError(f'cfg.data.{args.calib_split} not found')
    dataset = build_dataset(split_cfg, dict(test_mode=True))
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
        drop_last=False,
        dist=False,
        seed=cfg.get('seed', None),
    )
    data_loader = build_dataloader(dataset, **loader_cfg)

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_static

    sess = ort.InferenceSession(args.onnx_fp32, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    dr = _CalibReader(data_loader, input_name=input_name, max_batches=args.calib_batches)

    os.makedirs(os.path.dirname(args.onnx_int8), exist_ok=True)
    quantize_static(
        model_input=args.onnx_fp32,
        model_output=args.onnx_int8,
        calibration_data_reader=dr,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        optimize_model=True,
    )
    print(args.onnx_int8)


if __name__ == '__main__':
    main()

