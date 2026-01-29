#!/usr/bin/env python3
"""
医学图像格式转换工具

支持格式：
- npz序列：NumPy数组序列
- nii：NIfTI格式
- dcm序列：DICOM序列
- png序列：PNG图像序列

依赖：
- nibabel (pip install nibabel)
- pydicom (pip install pydicom)
- numpy
- pillow (pip install pillow)

用法：
    python image_converter.py -i input.npz -o output.nii -m npz2nii
    python image_converter.py  # 交互式模式

转换模式：
- npz2nii: npz 到 nii
- npz2dcm: npz 到 dcm序列
- npz2png: npz 到 png序列
- nii2npz: nii 到 npz
- nii2dcm: nii 到 dcm序列
- nii2png: nii 到 png序列
- dcm2npz: dcm序列 到 npz
- dcm2nii: dcm序列 到 nii
- dcm2png: dcm序列 到 png序列
- png2npz: png序列 到 npz
- png2nii: png序列 到 nii
- png2dcm: png序列 到 dcm序列
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import glob

try:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    import pydicom
    from pydicom.dataset import Dataset
    _DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"错误：缺少依赖库。运行 'pip install nibabel pydicom numpy pillow'")
    print(f"具体错误：{e}")
    sys.exit(1)

def load_npz(file_path: str) -> np.ndarray:
    """加载npz文件，返回数组"""
    data = np.load(file_path)
    # 假设数据在 'arr_0' 或其他键中
    if len(data.files) == 1:
        return data[data.files[0]]
    else:
        # 如果有多个数组，取第一个
        return data[data.files[0]]

def save_npz(data: np.ndarray, output_path: str):
    """保存为npz"""
    np.savez(output_path, data)

def load_nii(file_path: str) -> np.ndarray:
    """加载nii文件"""
    img = nib.load(file_path)
    return img.get_fdata()

def save_nii(data: np.ndarray, output_path: str, affine=None):
    """保存为nii"""
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)

def load_dcm_series(dir_path: str) -> np.ndarray:
    """加载dcm序列"""
    dcm_files = sorted(glob.glob(os.path.join(dir_path, "*.dcm")))
    if not dcm_files:
        raise ValueError("未找到DICOM文件")
    
    slices = []
    for dcm_file in dcm_files:
        ds = pydicom.dcmread(dcm_file)
        slices.append(ds.pixel_array)
    
    return np.stack(slices, axis=-1)  # 假设最后一维是切片

def save_dcm_series(data: np.ndarray, output_dir: str):
    """保存为dcm序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(data.shape[-1]):
        slice_data = data[..., i]
        ds = Dataset()
        # 设置基本DICOM属性
        ds.ImageType = ['ORIGINAL', 'PRIMARY']
        ds.SOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')  # CT Image Storage
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.Modality = 'CT'
        ds.Rows, ds.Columns = slice_data.shape[:2]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelData = slice_data.astype(np.uint16).tobytes()
        ds.InstanceNumber = i + 1
        
        output_path = os.path.join(output_dir, f"slice_{i+1:04d}.dcm")
        ds.save_as(output_path)

def load_png_series(dir_path: str) -> np.ndarray:
    """加载png序列"""
    png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not png_files:
        raise ValueError("未找到PNG文件")
    
    slices = []
    for png_file in png_files:
        img = Image.open(png_file)
        slices.append(np.array(img))
    
    return np.stack(slices, axis=-1)

def save_png_series(data: np.ndarray, output_dir: str):
    """保存为png序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(data.shape[-1]):
        slice_data = data[..., i]
        img = Image.fromarray(slice_data.astype(np.uint8))
        output_path = os.path.join(output_dir, f"slice_{i+1:04d}.png")
        img.save(output_path)

def convert(input_path: str, output_path: str, mode: str):
    """执行转换"""
    if mode == 'npz2nii':
        data = load_npz(input_path)
        save_nii(data, output_path)
    elif mode == 'npz2dcm':
        data = load_npz(input_path)
        save_dcm_series(data, output_path)
    elif mode == 'npz2png':
        data = load_npz(input_path)
        save_png_series(data, output_path)
    elif mode == 'nii2npz':
        data = load_nii(input_path)
        save_npz(data, output_path)
    elif mode == 'nii2dcm':
        data = load_nii(input_path)
        save_dcm_series(data, output_path)
    elif mode == 'nii2png':
        data = load_nii(input_path)
        save_png_series(data, output_path)
    elif mode == 'dcm2npz':
        data = load_dcm_series(input_path)
        save_npz(data, output_path)
    elif mode == 'dcm2nii':
        data = load_dcm_series(input_path)
        save_nii(data, output_path)
    elif mode == 'dcm2png':
        data = load_dcm_series(input_path)
        save_png_series(data, output_path)
    elif mode == 'png2npz':
        data = load_png_series(input_path)
        save_npz(data, output_path)
    elif mode == 'png2nii':
        data = load_png_series(input_path)
        save_nii(data, output_path)
    elif mode == 'png2dcm':
        data = load_png_series(input_path)
        save_dcm_series(data, output_path)
    else:
        raise ValueError(f"不支持的转换模式：{mode}")

def interactive_mode():
    """交互式模式"""
    print("=== 医学图像格式转换工具 ===")
    
    # 选择输入格式
    input_formats = ['npz', 'nii', 'dcm', 'png']
    print("选择输入格式：")
    for i, fmt in enumerate(input_formats):
        print(f"{i+1}. {fmt}")
    input_choice = int(input("请输入数字：")) - 1
    input_format = input_formats[input_choice]
    
    # 选择输出格式
    output_formats = [fmt for fmt in input_formats if fmt != input_format]
    print("选择输出格式：")
    for i, fmt in enumerate(output_formats):
        print(f"{i+1}. {fmt}")
    output_choice = int(input("请输入数字：")) - 1
    output_format = output_formats[output_choice]
    
    mode = f"{input_format}2{output_format}"
    
    # 输入路径
    if input_format == 'dcm' or input_format == 'png':
        input_path = input("输入文件夹路径：")
    else:
        input_path = input("输入文件路径：")
    
    # 输出路径
    if output_format == 'dcm' or output_format == 'png':
        output_path = input("输出文件夹路径：")
    else:
        output_path = input("输出文件路径：")
    
    try:
        convert(input_path, output_path, mode)
        print("转换完成！")
    except Exception as e:
        print(f"转换失败：{e}")

def main():
    parser = argparse.ArgumentParser(description="医学图像格式转换工具")
    parser.add_argument('-i', '--input', help='输入文件/文件夹路径')
    parser.add_argument('-o', '--output', help='输出文件/文件夹路径')
    parser.add_argument('-m', '--mode', help='转换模式，如 npz2nii')
    
    args = parser.parse_args()
    
    if args.input and args.output and args.mode:
        try:
            convert(args.input, args.output, args.mode)
            print("转换完成！")
        except Exception as e:
            print(f"转换失败：{e}")
            sys.exit(1)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()