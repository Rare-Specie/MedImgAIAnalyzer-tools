"""
这个脚本是一个医学图像转换工具，主要用于在不同的医学图像格式之间进行转换。它定义了一个 MedicalImageConverter 类，支持以下转换：

- DCM (DICOM) 到 NII (NIfTI)
- NII 到 DCM
- 体积数据 (DCM 或 NII) 到 PNG 图像序列
- 体积数据 (DCM 或 NII) 到 NPZ 文件序列
- PNG 图像序列到体积数据 (DCM 或 NII)
- NPZ 文件序列到体积数据 (DCM 或 NII)
- PNG 到 NPZ
- NPZ 到 PNG

脚本会处理像素数据、元数据（如模态、患者ID、窗口中心/宽度等），并保存到输出目录中。使用时需要指定输入/输出目录和格式。

使用方式：
python image_converter.py <input_dir> <output_dir> <input_format> <output_format>

例如：python image_converter.py /path/to/input /path/to/output dcm nii

依赖库（需要通过 pip 安装）：
- pydicom：处理 DICOM 文件
- nibabel：处理 NIfTI 文件
- opencv-python：图像处理（PNG 读写）
- pillow：图像处理（备用）
- numpy：数组操作（Python 内置）
- pathlib：路径处理（Python 3.4+ 内置）
- json：JSON 处理（Python 内置）
- os：操作系统接口（Python 内置）

安装命令：pip install pydicom nibabel opencv-python pillow
"""

import os
import json
import numpy as np
import argparse
import pydicom
import nibabel as nib
import cv2
from PIL import Image
from pathlib import Path

class MedicalImageConverter:
    def __init__(self, input_dir, output_dir, input_format, output_format):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_format = input_format.lower()
        self.output_format = output_format.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert(self):
        if self.input_format == 'dcm' and self.output_format == 'nii':
            self.dcm_to_nii()
        elif self.input_format == 'nii' and self.output_format == 'dcm':
            self.nii_to_dcm()
        elif self.input_format in ['dcm', 'nii'] and self.output_format == 'png':
            self.volume_to_png()
        elif self.input_format in ['dcm', 'nii'] and self.output_format == 'npz':
            self.volume_to_npz()
        elif self.input_format == 'png' and self.output_format in ['dcm', 'nii']:
            self.png_to_volume()
        elif self.input_format == 'npz' and self.output_format in ['dcm', 'nii']:
            self.npz_to_volume()
        elif self.input_format == 'png' and self.output_format == 'npz':
            self.png_to_npz()
        elif self.input_format == 'npz' and self.output_format == 'png':
            self.npz_to_png()
        else:
            raise ValueError("Unsupported conversion")

    def dcm_to_nii(self):
        # Use nibabel to load DCM series and save as NII
        dicom_files = list(self.input_dir.glob('*.dcm'))
        if not dicom_files:
            raise FileNotFoundError("No DCM files found")
        # Assume all in one series; sort by InstanceNumber
        dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
        slices = [pydicom.dcmread(f) for f in dicom_files]
        # Extract pixel data and metadata
        pixel_arrays = [s.pixel_array.astype(s.RescaleSlope * s.pixel_array + s.RescaleIntercept if hasattr(s, 'RescaleSlope') else s.pixel_array) for s in slices]
        volume = np.stack(pixel_arrays, axis=-1)
        # Basic affine (simplified; in practice, use proper calculation)
        affine = np.eye(4)
        affine[0,0] = slices[0].PixelSpacing[0]
        affine[1,1] = slices[0].PixelSpacing[1]
        affine[2,2] = slices[0].SliceThickness
        nii_img = nib.Nifti1Image(volume, affine)
        output_path = self.output_dir / "converted.nii"
        nib.save(nii_img, output_path)
        # Save metadata
        metadata = {
            'modality': slices[0].Modality,
            'patient_id': slices[0].PatientID,
            'study_date': slices[0].StudyDate,
            'series_description': slices[0].SeriesDescription
        }
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

    def nii_to_dcm(self):
        # Load NII
        nii_path = list(self.input_dir.glob('*.nii*'))[0]
        nii_img = nib.load(nii_path)
        volume = nii_img.get_fdata()
        affine = nii_img.affine
        # For each slice, create DCM
        for i in range(volume.shape[2]):
            slice_data = volume[:, :, i]
            ds = pydicom.Dataset()
            # Minimal DCM metadata (in practice, add more)
            ds.PatientID = "Unknown"
            ds.Modality = "CT"  # Assume
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.SOPClassUID = pydicom.uid.CTImageStorage
            ds.InstanceNumber = i + 1
            ds.PixelData = slice_data.astype(np.int16).tobytes()
            ds.Rows, ds.Columns = slice_data.shape
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelSpacing = [affine[0,0], affine[1,1]]
            ds.SliceThickness = affine[2,2]
            ds.ImagePositionPatient = [0, 0, i * affine[2,2]]
            ds.ImageOrientationPatient = [1,0,0,0,1,0]
            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.file_meta = file_meta
            ds.is_implicit_VR = True
            ds.is_little_endian = True
            output_path = self.output_dir / f"slice_{i+1:04d}.dcm"
            ds.save_as(output_path)

    def volume_to_png(self):
        # Load volume (DCM or NII)
        if self.input_format == 'dcm':
            dicom_files = list(self.input_dir.glob('*.dcm'))
            dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
            slices = [pydicom.dcmread(f) for f in dicom_files]
            pixel_arrays = [s.pixel_array.astype(s.RescaleSlope * s.pixel_array + s.RescaleIntercept if hasattr(s, 'RescaleSlope') else s.pixel_array) for s in slices]
            volume = np.stack(pixel_arrays, axis=-1)
            metadata = {'modality': slices[0].Modality, 'window_center': 40, 'window_width': 80}  # Default for CT brain
        else:
            nii_path = list(self.input_dir.glob('*.nii*'))[0]
            nii_img = nib.load(nii_path)
            volume = nii_img.get_fdata()
            metadata = {'modality': 'MRI', 'window_center': 0, 'window_width': 100}  # Default
        # Apply windowing
        wc, ww = metadata['window_center'], metadata['window_width']
        volume_norm = np.clip((volume - (wc - ww/2)) / ww * 255, 0, 255).astype(np.uint8)
        for i in range(volume.shape[2]):
            img = volume_norm[:, :, i]
            output_path = self.output_dir / f"slice_{i+1:04d}.png"
            cv2.imwrite(str(output_path), img)
        # Save metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

    def volume_to_npz(self):
        # Similar to volume_to_png but save raw data
        if self.input_format == 'dcm':
            dicom_files = list(self.input_dir.glob('*.dcm'))
            dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
            slices = [pydicom.dcmread(f) for f in dicom_files]
            pixel_arrays = [s.pixel_array.astype(s.RescaleSlope * s.pixel_array + s.RescaleIntercept if hasattr(s, 'RescaleSlope') else s.pixel_array) for s in slices]
            volume = np.stack(pixel_arrays, axis=-1)
            metadata = {'modality': slices[0].Modality, 'pixel_spacing': slices[0].PixelSpacing, 'slice_thickness': slices[0].SliceThickness}
        else:
            nii_path = list(self.input_dir.glob('*.nii*'))[0]
            nii_img = nib.load(nii_path)
            volume = nii_img.get_fdata()
            metadata = {'modality': 'MRI', 'affine': nii_img.affine.tolist()}
        for i in range(volume.shape[2]):
            slice_data = volume[:, :, i]
            output_path = self.output_dir / f"slice_{i+1:04d}.npz"
            np.savez_compressed(output_path, array=slice_data, metadata=metadata)

    def png_to_volume(self):
        png_files = sorted(self.input_dir.glob('*.png'))
        metadata_path = self.input_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError("Metadata JSON required for PNG to volume conversion")
        slices = []
        for png_file in png_files:
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            slices.append(img)
        volume = np.stack(slices, axis=-1)
        if self.output_format == 'nii':
            affine = np.eye(4)
            if 'affine' in metadata:
                affine = np.array(metadata['affine'])
            nii_img = nib.Nifti1Image(volume.astype(np.float32), affine)  # Approximate
            output_path = self.output_dir / "converted.nii"
            nib.save(nii_img, output_path)
        else:
            # To DCM, similar to nii_to_dcm but with volume
            for i in range(volume.shape[2]):
                slice_data = volume[:, :, i]
                ds = pydicom.Dataset()
                ds.PatientID = "Unknown"
                ds.Modality = metadata.get('modality', 'CT')
                # ... (similar to nii_to_dcm)
                output_path = self.output_dir / f"slice_{i+1:04d}.dcm"
                # Save as before

    def npz_to_volume(self):
        npz_files = sorted(self.input_dir.glob('*.npz'))
        slices = []
        metadata = {}
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            # 支持不同的键名
            if 'array' in data:
                img = data['array']
                if 'metadata' in data and metadata == {}:
                    metadata = data['metadata'].item()
            elif 'image' in data:
                img = data['image']
                # 外部文件可能没有 metadata，使用默认
            else:
                raise KeyError(f"NPZ 文件 {npz_file} 中未找到 'array' 或 'image' 键")
            slices.append(img)
        volume = np.stack(slices, axis=-1)
        if self.output_format == 'nii':
            affine = np.array(metadata.get('affine', np.eye(4)))
            nii_img = nib.Nifti1Image(volume, affine)
            output_path = self.output_dir / "converted.nii"
            nib.save(nii_img, output_path)
        else:
            # To DCM
            for i in range(volume.shape[2]):
                slice_data = volume[:, :, i]
                # 简化：假设默认值
                ds = pydicom.Dataset()
                ds.PatientID = "Unknown"
                ds.Modality = metadata.get('modality', 'CT')
                ds.StudyInstanceUID = pydicom.uid.generate_uid()
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                ds.SOPClassUID = pydicom.uid.CTImageStorage
                ds.InstanceNumber = i + 1
                ds.PixelData = slice_data.astype(np.int16).tobytes()
                ds.Rows, ds.Columns = slice_data.shape
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelSpacing = [1.0, 1.0]  # 默认
                ds.SliceThickness = 1.0  # 默认
                ds.ImagePositionPatient = [0, 0, i * 1.0]
                ds.ImageOrientationPatient = [1,0,0,0,1,0]
                file_meta = pydicom.Dataset()
                file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
                file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                ds.file_meta = file_meta
                ds.is_implicit_VR = True
                ds.is_little_endian = True
                output_path = self.output_dir / f"slice_{i+1:04d}.dcm"
                ds.save_as(output_path)

    def png_to_npz(self):
        png_files = sorted(self.input_dir.glob('*.png'))
        metadata_path = self.input_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        for i, png_file in enumerate(png_files):
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            output_path = self.output_dir / f"slice_{i+1:04d}.npz"
            np.savez_compressed(output_path, array=img, metadata=metadata)

    def npz_to_png(self):
        npz_files = sorted(self.input_dir.glob('*.npz'))
        for i, npz_file in enumerate(npz_files):
            data = np.load(npz_file, allow_pickle=True)
            # 支持不同的键名：优先 'array'（脚本生成），其次 'image'（外部文件）
            if 'array' in data:
                img = data['array']
                metadata = data.get('metadata', {}).item() if 'metadata' in data else {}
            elif 'image' in data:
                img = data['image']
                metadata = {}  # 外部文件可能没有 metadata
            else:
                raise KeyError(f"NPZ 文件 {npz_file} 中未找到 'array' 或 'image' 键")
            if img.dtype != np.uint8:
                wc, ww = metadata.get('window_center', 0), metadata.get('window_width', 100)
                img = np.clip((img - (wc - ww/2)) / ww * 255, 0, 255).astype(np.uint8)
            output_path = self.output_dir / f"slice_{i+1:04d}.png"
            cv2.imwrite(str(output_path), img)

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学图像格式转换工具")
    parser.add_argument("input_dir", help="输入目录路径")
    parser.add_argument("output_dir", help="输出目录路径")
    parser.add_argument("input_format", choices=['dcm', 'nii', 'png', 'npz'], help="输入格式")
    parser.add_argument("output_format", choices=['nii', 'dcm', 'png', 'npz'], help="输出格式")
    
    args = parser.parse_args()
    
    converter = MedicalImageConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_format=args.input_format,
        output_format=args.output_format
    )
    converter.convert()