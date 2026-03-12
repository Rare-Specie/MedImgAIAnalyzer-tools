#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 NPZ 中的 image.npy 做常见图像处理：缩放、旋转、裁切、对比度调整、伽马调整。
仅修改 image.npy，其他数组键保持不变。

示例：
python pynpzprocess/npz_image_processor.py \
  --input "/path/to/case.npz" \
  --scale-x 1.2 --scale-y 1.2 \
  --rotate 15 \
  --crop 10 20 256 256 \
  --contrast 1.15 \
  --gamma 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "未检测到 opencv-python，请先安装：pip install opencv-python\n"
        f"原始错误：{exc}"
    )


ArrayLike = np.ndarray


def _clip_crop_rect(x: int, y: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 <= x:
        x, x2 = 0, max(1, width)
    if y2 <= y:
        y, y2 = 0, max(1, height)
    return x, y, x2 - x, y2 - y


def _adjust_gamma(img: ArrayLike, gamma: float) -> ArrayLike:
    if gamma <= 0:
        raise ValueError("gamma 必须大于 0")

    orig_dtype = img.dtype
    work = img.astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        max_val = float(np.iinfo(orig_dtype).max)
        if max_val <= 0:
            return img
        norm = np.clip(work / max_val, 0.0, 1.0)
        out = np.power(norm, gamma) * max_val
    else:
        min_v = float(np.min(work))
        max_v = float(np.max(work))
        if max_v <= min_v:
            return img
        if min_v >= 0.0 and max_v <= 1.0:
            out = np.power(np.clip(work, 0.0, 1.0), gamma)
        else:
            norm = (work - min_v) / (max_v - min_v)
            out = np.power(np.clip(norm, 0.0, 1.0), gamma) * (max_v - min_v) + min_v

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(orig_dtype)


def _process_2d_or_hwc(
    image: ArrayLike,
    scale_x: float,
    scale_y: float,
    rotate_deg: float,
    crop: Tuple[int, int, int, int] | None,
    contrast: float,
    gamma: float,
) -> ArrayLike:
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale-x 和 scale-y 必须大于 0")

    orig_dtype = image.dtype
    out = image.astype(np.float32)

    if scale_x != 1.0 or scale_y != 1.0:
        out = cv2.resize(out, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    if rotate_deg != 0.0:
        h, w = out.shape[:2]
        center = (w / 2.0, h / 2.0)
        mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        out = cv2.warpAffine(
            out,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    if crop is not None:
        x, y, cw, ch = _clip_crop_rect(*crop, width=out.shape[1], height=out.shape[0])
        out = out[y : y + ch, x : x + cw, ...]

    if contrast != 1.0:
        out = out * contrast

    if gamma != 1.0:
        out = _adjust_gamma(out, gamma).astype(np.float32)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(out, info.min, info.max)

    return out.astype(orig_dtype)


def process_image(
    image: ArrayLike,
    scale_x: float,
    scale_y: float,
    rotate_deg: float,
    crop: Tuple[int, int, int, int] | None,
    contrast: float,
    gamma: float,
    preserve_resolution: bool = False,
    orig_shape: Tuple[int, int] | None = None,
) -> ArrayLike:
    if image.ndim == 2:
        out = _process_2d_or_hwc(image, scale_x, scale_y, rotate_deg, crop, contrast, gamma)
        if preserve_resolution and orig_shape is not None and out.shape != orig_shape:
            # 若执行了裁切并要求恢复分辨率，使用缩放将裁切区域放大填满原始分辨率；
            # 否则重采样回原始大小
            out = _resize_to_shape(out, orig_shape, interp=cv2.INTER_LINEAR)
        return out

    if image.ndim == 3:
        # HWC 彩色图
        if image.shape[-1] in (1, 3, 4):
            out = _process_2d_or_hwc(image, scale_x, scale_y, rotate_deg, crop, contrast, gamma)
            if preserve_resolution and orig_shape is not None and out.shape[:2] != orig_shape:
                if crop is not None:
                    out = _resize_to_shape(out, orig_shape, interp=cv2.INTER_LINEAR)
                else:
                    out = _resize_to_shape(out, orig_shape, interp=cv2.INTER_LINEAR)
            return out

        # CHW 彩色图
        if image.shape[0] in (1, 3, 4):
            hwc = np.transpose(image, (1, 2, 0))
            proc = _process_2d_or_hwc(hwc, scale_x, scale_y, rotate_deg, crop, contrast, gamma)
            if preserve_resolution and orig_shape is not None and proc.shape[:2] != orig_shape:
                if crop is not None:
                    proc = _resize_to_shape(proc, orig_shape, interp=cv2.INTER_LINEAR)
                else:
                    proc = _resize_to_shape(proc, orig_shape, interp=cv2.INTER_LINEAR)
            return np.transpose(proc, (2, 0, 1))

        # 默认按 (N, H, W) 逐切片处理
        slices = [
            _process_2d_or_hwc(s, scale_x, scale_y, rotate_deg, crop, contrast, gamma)
            for s in image
        ]
        stacked = np.stack(slices, axis=0)
        if preserve_resolution and orig_shape is not None and stacked.shape[1:] != orig_shape:
            # resize each slice back
            resized = [
                _resize_to_shape(s, orig_shape, interp=cv2.INTER_LINEAR) for s in stacked
            ]
            return np.stack(resized, axis=0)
        return stacked

    raise ValueError(f"不支持的 image 维度: {image.ndim}，仅支持 2D 或 3D")


def _process_label(
    label: ArrayLike,
    scale_x: float,
    scale_y: float,
    rotate_deg: float,
    crop: Tuple[int, int, int, int] | None,
) -> ArrayLike:
    # 对 label 仅做几何变换，使用最近邻插值以保持标签值
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale-x 和 scale-y 必须大于 0")

    orig_dtype = label.dtype
    out = label.astype(np.float32)

    if scale_x != 1.0 or scale_y != 1.0:
        # 对单通道或多通道同样处理
        if out.ndim == 2:
            out = cv2.resize(out, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
        elif out.ndim == 3 and out.shape[-1] in (1, 3, 4):
            out = cv2.resize(out, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
        elif out.ndim == 3 and out.shape[0] in (1, 3, 4):
            hwc = np.transpose(out, (1, 2, 0))
            hwc = cv2.resize(hwc, dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            out = np.transpose(hwc, (2, 0, 1))
        else:
            # treat as stack of slices
            slices = [
                cv2.resize(s.astype(np.float32), dsize=None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                for s in out
            ]
            out = np.stack(slices, axis=0)

    if rotate_deg != 0.0:
        if out.ndim == 2:
            h, w = out.shape[:2]
            center = (w / 2.0, h / 2.0)
            mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
            out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        elif out.ndim == 3 and out.shape[-1] in (1, 3, 4):
            h, w = out.shape[:2]
            center = (w / 2.0, h / 2.0)
            mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
            out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        elif out.ndim == 3 and out.shape[0] in (1, 3, 4):
            hwc = np.transpose(out, (1, 2, 0))
            h, w = hwc.shape[:2]
            center = (w / 2.0, h / 2.0)
            mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
            hwc = cv2.warpAffine(hwc, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            out = np.transpose(hwc, (2, 0, 1))
        else:
            slices = []
            for s in out:
                h, w = s.shape[:2]
                center = (w / 2.0, h / 2.0)
                mat = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
                slices.append(cv2.warpAffine(s, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT))
            out = np.stack(slices, axis=0)

    if crop is not None:
        x, y, cw, ch = _clip_crop_rect(*crop, width=out.shape[1], height=out.shape[0])
        out = out[y : y + ch, x : x + cw, ...]

    return out.astype(orig_dtype)


def _resize_to_shape(arr: ArrayLike, target_shape: Tuple[int, ...], interp: int) -> ArrayLike:
    # target_shape: (H, W) or (H, W, C) or (C, H, W)
    if arr.ndim == 2:
        h, w = target_shape[0], target_shape[1]
        return cv2.resize(arr, (w, h), interpolation=interp)
    if arr.ndim == 3:
        # HWC
        if arr.shape[-1] in (1, 3, 4):
            h, w = target_shape[0], target_shape[1]
            return cv2.resize(arr, (w, h), interpolation=interp)
        # CHW -> transpose
        if arr.shape[0] in (1, 3, 4):
            hwc = np.transpose(arr, (1, 2, 0))
            h, w = target_shape[0], target_shape[1]
            hwc = cv2.resize(hwc, (w, h), interpolation=interp)
            return np.transpose(hwc, (2, 0, 1))
        # stack of slices
        slices = [cv2.resize(s, (target_shape[1], target_shape[0]), interpolation=interp) for s in arr]
        return np.stack(slices, axis=0)
    return arr


def _derive_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return input_path.with_name(f"{input_path.stem}_processed.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 NPZ 中的 image.npy 做图像处理（仅修改 image 键）")
    parser.add_argument("--input", required=True, help="输入 npz 文件路径")
    parser.add_argument("--output", default=None, help="输出 npz 文件路径，默认 *_processed.npz")
    parser.add_argument("--scale-x", type=float, default=1.0, help="X 方向缩放倍率")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Y 方向缩放倍率")
    parser.add_argument("--rotate", type=float, default=0.0, help="旋转角度（度，逆时针）")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="裁切区域，格式: X Y W H",
    )
    parser.add_argument("--contrast", type=float, default=1.0, help="对比度倍率，1.0 为不变")
    parser.add_argument("--gamma", type=float, default=1.0, help="伽马值，1.0 为不变")
    parser.add_argument(
        "--preserve-resolution",
        action="store_true",
        help="处理后将结果恢复为输入的原始分辨率（几何变换后重采样或将裁切放回原位）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    output_path = _derive_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path, allow_pickle=False) as data:
        keys = list(data.files)
        if "image" not in data:
            raise SystemExit(f"输入 npz 不包含 image 键: {input_path}")

        orig_shape = data["image"].shape

        processed = process_image(
            data["image"],
            scale_x=args.scale_x,
            scale_y=args.scale_y,
            rotate_deg=args.rotate,
            crop=tuple(args.crop) if args.crop is not None else None,
            contrast=args.contrast,
            gamma=args.gamma,
            preserve_resolution=args.preserve_resolution,
            orig_shape=orig_shape,
        )

        save_dict = {k: data[k] for k in keys}
        save_dict["image"] = processed

        # 如果存在 label 并且有几何变换（scale/rotate/crop），对 label 做相同的几何变换
        if "label" in data:
            geom_changed = (args.scale_x != 1.0) or (args.scale_y != 1.0) or (args.rotate != 0.0) or (args.crop is not None)
            if geom_changed:
                try:
                    processed_label = _process_label(
                        data["label"],
                        scale_x=args.scale_x,
                        scale_y=args.scale_y,
                        rotate_deg=args.rotate,
                        crop=tuple(args.crop) if args.crop is not None else None,
                    )
                    # 若要求保留分辨率，需要把 label 恢复到原始形状
                    if args.preserve_resolution and processed_label.shape != orig_shape:
                        # 若是裁切操作且 crop 给出，则把裁切结果放回原位
                        if args.crop is not None:
                            canvas = np.zeros(orig_shape, dtype=processed_label.dtype)
                            x, y, cw, ch = _clip_crop_rect(*tuple(args.crop), width=orig_shape[1], height=orig_shape[0])
                            canvas[y : y + ch, x : x + cw, ...] = processed_label
                            processed_label = canvas
                        else:
                            processed_label = _resize_to_shape(processed_label, orig_shape, interp=cv2.INTER_NEAREST)

                    save_dict["label"] = processed_label
                except Exception:
                    # 若处理失败则保留原始 label
                    pass

    np.savez_compressed(output_path, **save_dict)
    print(f"处理完成: {output_path}")
    print(f"保留原始键: {', '.join(keys)}")
    print(f"image 形状: {save_dict['image'].shape}, dtype: {save_dict['image'].dtype}")


if __name__ == "__main__":
    main()
