#!/usr/bin/env python3
"""Convert between NIfTI, NPZ, PNG series and DICOM folders.

Supports CLI arguments for batch usage and an interactive prompt when arguments are omitted.
"""
# 依赖: numpy, nibabel (处理 NIfTI), opencv-python (PNG 编解码), pydicom (DICOM I/O).
# 使用示例:
#   python image_converter.py <src_dir> <dst_dir> <src_fmt> <dst_fmt> --slice-axis 2 --png-prefix png_ --png-ext png
#   python image_converter.py /path/to/dicom_dir /tmp/output_dir dcm nii --reference-nii /path/to/template.nii
#   python image_converter.py  # 进入交互模式并按提示操作。
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - user will install if needed
    nib = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover
    pydicom = None
    Dataset = None
    FileDataset = None
    ExplicitVRLittleEndian = None
    generate_uid = None


SUPPORTED_FORMATS = ("nii", "npz", "png", "dcm")
FORMAT_ALIASES = {
    "nifti": "nii",
    "dicom": "dcm",
    "npy": "npz",
}
NPZ_META_KEYS = {"affine", "spacing", "source_type", "source_name"}


@dataclass
class Volume:
    data: np.ndarray
    affine: np.ndarray
    spacing: Tuple[float, float, float]
    source_type: str
    source_name: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim > 3:
            arr = arr[..., 0]
        self.data = arr
        self.affine = ensure_affine(self.affine)
        self.spacing = ensure_spacing(self.spacing)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)


# --- helpers ---

def ensure_affine(matrix: Optional[np.ndarray]) -> np.ndarray:
    if matrix is None:
        return np.eye(4, dtype=np.float32)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    base = np.eye(4, dtype=np.float32)
    rows, cols = arr.shape
    base[:rows, :cols] = arr
    return base


def ensure_spacing(spacing: Sequence[float]) -> Tuple[float, float, float]:
    if spacing is None:
        return 1.0, 1.0, 1.0
    vals = list(spacing)
    while len(vals) < 3:
        vals.append(vals[-1] if vals else 1.0)
    return tuple(float(vals[i]) for i in range(3))


def natural_sort_key(value: str) -> Iterable:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def require_nibabel() -> None:
    if nib is None:
        raise RuntimeError("Install nibabel (pip install nibabel) to convert NIfTI files")


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError("Install opencv-python (pip install opencv-python) to read/write PNG series")


def require_pydicom() -> None:
    if pydicom is None or Dataset is None or FileDataset is None or ExplicitVRLittleEndian is None:
        raise RuntimeError("Install pydicom (pip install pydicom) to read/write DICOM files")


def normalize_to_uint8(slice_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_data, dtype=np.float32)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if high > low:
        scaled = (arr - low) / (high - low) * 255.0
    else:
        scaled = np.zeros_like(arr, dtype=np.float32)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def extract_nii_stem(path: str) -> str:
    name = os.path.basename(path)
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def ensure_dir_exists(path: str) -> None:
    if not os.path.isdir(path):
        raise ValueError(f"输入路径必须是文件夹: {path}")


# --- format readers ---

def load_nii(path: str) -> Volume:
    require_nibabel()
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header
    spacing = tuple(float(z) for z in header.get_zooms()[:3])
    return Volume(data, affine, spacing, "nii", os.path.basename(path))


def load_npz_items(path: str) -> list[tuple[Volume, str]]:
    with np.load(path, allow_pickle=True) as archive:
        arrays = []
        for key in archive.files:
            if key in NPZ_META_KEYS:
                continue
            value = archive[key]
            if isinstance(value, np.ndarray):
                arrays.append((key, value))
        if not arrays:
            candidates = ["image", "img", "data", "volume", "slice"]
            array = None
            for key in candidates:
                if key in archive:
                    array = archive[key]
                    break
            if array is None:
                raise ValueError("NPZ archive does not contain arrays")
            arrays = [("volume", array)]
        spacing = archive.get("spacing")
        affine = archive.get("affine")
        if affine is None and "affine" in archive.files:
            affine = archive["affine"]
        items = []
        for key, array in arrays:
            if spacing is None and array.ndim >= 3:
                inferred_spacing = (1.0, 1.0, 1.0)
            else:
                inferred_spacing = spacing or (1.0, 1.0, 1.0)
            items.append(
                (
                    Volume(array, affine, inferred_spacing, "npz", os.path.basename(path)),
                    key,
                )
            )
        return items


def _stack_png_list(file_paths: Sequence[str], axis: int) -> np.ndarray:
    require_opencv()
    slices = []
    for path in file_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"无法读取PNG: {path}")
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        slices.append(img.astype(np.float32))
    stack = np.stack(slices, axis=0)
    if axis == 0:
        return stack
    if axis == 1:
        return np.transpose(stack, (1, 0, 2))
    return np.transpose(stack, (1, 2, 0))


def load_png_series(folder: str, axis: int, spacing: Optional[Sequence[float]]) -> Volume:
    require_opencv()
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"PNG folder not found: {folder}")
    entries = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not entries:
        raise ValueError("PNG folder does not contain .png files")
    entries.sort(key=natural_sort_key)
    full_paths = [os.path.join(folder, f) for f in entries]
    data = _stack_png_list(full_paths, axis)
    return Volume(data, np.eye(4), spacing or (1.0, 1.0, 1.0), "png", os.path.basename(folder))


def load_png_slices(folder: str, spacing: Optional[Sequence[float]]) -> list[tuple[Volume, str]]:
    require_opencv()
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"PNG folder not found: {folder}")
    entries = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not entries:
        raise ValueError("PNG folder does not contain .png files")
    entries.sort(key=natural_sort_key)
    items: list[tuple[Volume, str]] = []
    for filename in entries:
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"无法读取PNG: {path}")
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        name = os.path.splitext(filename)[0]
        items.append((Volume(img.astype(np.float32), np.eye(4), spacing or (1.0, 1.0, 1.0), "png", name), name))
    return items


def load_dicom_series(
    folder: str, reference_affine: Optional[np.ndarray], override_spacing: Optional[Sequence[float]]
) -> Volume:
    require_pydicom()
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".dcm")]
    if not files:
        raise ValueError(f"DICOM folder does not contain .dcm files: {folder}")
    datasets = [pydicom.dcmread(p, force=True) for p in files]
    datasets.sort(
        key=lambda ds: (
            float(getattr(ds, "SliceLocation", 0.0)),
            float(getattr(ds, "InstanceNumber", 0)),
            getattr(ds, "SOPInstanceUID", ""),
        )
    )
    stack = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=-1)
    ds0 = datasets[0]
    raw_spacing = (
        float(ds0.PixelSpacing[0]) if hasattr(ds0, "PixelSpacing") else 1.0,
        float(ds0.PixelSpacing[1]) if hasattr(ds0, "PixelSpacing") else 1.0,
        float(getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 1.0))),
    )
    spacing = tuple(override_spacing) if override_spacing is not None else raw_spacing
    affine = reference_affine if reference_affine is not None else np.diag((*spacing[:3], 1.0))
    return Volume(stack, affine, spacing, "dcm", os.path.basename(folder))


# --- format writers ---

def save_nii(volume: Volume, path: str) -> None:
    require_nibabel()
    ensure_dir(os.path.dirname(path) or ".")
    img = nib.Nifti1Image(volume.data, volume.affine)
    nib.save(img, path)


def save_npz(
    volume: Volume,
    path: str,
    data_override: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None,
    fortran_order: bool = False,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    data = data_override if data_override is not None else volume.data
    if dtype is not None:
        data = data.astype(dtype)
    if fortran_order:
        data = np.asfortranarray(data)
    np.savez_compressed(
        path,
        image=data,
        affine=volume.affine,
        spacing=np.array(volume.spacing, dtype=np.float32),
        source_type=volume.source_type,
        source_name=volume.source_name,
    )


def save_png_series(
    volume: Volume,
    folder: str,
    axis: int,
    prefix: str,
    ext: str,
) -> None:
    require_opencv()
    axis = axis % 3
    os.makedirs(folder, exist_ok=True)
    data = np.moveaxis(volume.data, axis, -1)
    for idx in range(data.shape[-1]):
        slice_img = normalize_to_uint8(data[..., idx])
        fname = f"{prefix}{idx + 1:03d}.{ext}"
        path = os.path.join(folder, fname)
        cv2.imwrite(path, slice_img)


def _create_base_dicom(dataset: Dataset, slice_data: np.ndarray, pixel_spacing: Tuple[float, float], slice_thickness: float) -> None:
    dataset.Rows, dataset.Columns = slice_data.shape
    dataset.PixelSpacing = list(pixel_spacing)
    dataset.SliceThickness = float(slice_thickness)
    dataset.SpacingBetweenSlices = float(slice_thickness)
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"


def save_dcm_series(
    volume: Volume,
    folder: str,
    patient_name: str,
    modality: str,
    study_description: str,
    series_description: str,
    pixel_spacing: Tuple[float, float, float],
    filename_prefix: str,
) -> None:
    require_pydicom()
    os.makedirs(folder, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    for idx in range(volume.data.shape[2]):
        ds = Dataset()
        ds.PatientName = patient_name
        ds.PatientID = "ANON"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.InstanceNumber = idx + 1
        ds.Modality = modality
        ds.SeriesDescription = series_description
        ds.StudyDescription = study_description
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
        ds.FrameOfReferenceUID = generate_uid()
        _create_base_dicom(ds, volume.data[:, :, idx], pixel_spacing[:2], pixel_spacing[2])
        ds.ImagePositionPatient = [0.0, 0.0, float(idx) * pixel_spacing[2]]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelData = volume.data[:, :, idx].astype(np.int16).tobytes()
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = file_meta
        file_path = os.path.join(folder, f"{filename_prefix}{idx + 1:03d}.dcm")
        file_ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        file_ds.update(ds)
        file_ds.is_little_endian = True
        file_ds.is_implicit_VR = False
        file_ds.save_as(file_path, write_like_original=False)


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# --- general conversion ---

def read_volumes(
    path: str,
    fmt: str,
    target_fmt: str,
    slice_axis: int,
    pixel_spacing: Optional[Sequence[float]],
    reference_nii: Optional[str],
) -> list[tuple[Volume, str]]:
    fmt = canonical_format(fmt)
    if fmt == "nii":
        volume = load_nii(path)
        return [(volume, extract_nii_stem(path))]
    if fmt == "npz":
        ensure_dir_exists(path)
        files = [f for f in os.listdir(path) if f.lower().endswith(".npz")]
        if not files:
            raise ValueError(f"文件夹中未找到 npz 文件: {path}")
        files.sort(key=natural_sort_key)
        items: list[tuple[Volume, str]] = []
        for filename in files:
            npz_path = os.path.join(path, filename)
            base = os.path.splitext(filename)[0]
            for volume, key in load_npz_items(npz_path):
                items.append((volume, f"{base}_{key}"))
        return items
    if fmt == "png":
        ensure_dir_exists(path)
        if canonical_format(target_fmt) == "npz":
            return load_png_slices(path, pixel_spacing)
        volume = load_png_series(path, slice_axis, pixel_spacing)
        return [(volume, os.path.basename(path))]
    if fmt == "dcm":
        ensure_dir_exists(path)
        ref_affine = None
        if reference_nii:
            ref_affine = load_nii(reference_nii).affine
        volume = load_dicom_series(path, ref_affine, pixel_spacing)
        return [(volume, os.path.basename(path))]
    raise ValueError(f"Unsupported source format: {fmt}")


def is_file_path(path: str, exts: Sequence[str]) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in exts)


def write_volumes(
    volumes: list[tuple[Volume, str]],
    output_path: str,
    fmt: str,
    slice_axis: int,
    pixel_spacing: Optional[Sequence[float]],
    png_prefix: str,
    png_ext: str,
    dcm_patient: str,
    dcm_modality: str,
    dcm_series_desc: str,
    dcm_study_desc: str,
) -> None:
    fmt = canonical_format(fmt)
    multi = len(volumes) > 1
    if fmt == "nii":
        if not multi and is_file_path(output_path, (".nii", ".nii.gz")):
            save_nii(volumes[0][0], output_path)
            return
        os.makedirs(output_path, exist_ok=True)
        for volume, name in volumes:
            save_nii(volume, os.path.join(output_path, f"{name}.nii"))
        return
    if fmt == "npz":
        if not multi and is_file_path(output_path, (".npz",)):
            volume = volumes[0][0]
            if volume.source_type == "png" and volume.data.shape[2] == 1:
                data_2d = volume.data[:, :, 0] / 255.0
                save_npz(volume, output_path, data_override=data_2d, dtype=np.float64, fortran_order=True)
            else:
                save_npz(volume, output_path)
            return
        os.makedirs(output_path, exist_ok=True)
        for volume, name in volumes:
            out_path = os.path.join(output_path, f"{name}.npz")
            if volume.source_type == "png" and volume.data.shape[2] == 1:
                data_2d = volume.data[:, :, 0] / 255.0
                save_npz(volume, out_path, data_override=data_2d, dtype=np.float64, fortran_order=True)
            else:
                save_npz(volume, out_path)
        return
    if fmt == "png":
        os.makedirs(output_path, exist_ok=True)
        for volume, name in volumes:
            prefix = f"{name}_" + (png_prefix or "")
            save_png_series(volume, output_path, slice_axis, prefix, png_ext)
        return
    if fmt == "dcm":
        os.makedirs(output_path, exist_ok=True)
        for volume, name in volumes:
            prefix = f"{name}_"
            save_dcm_series(
                volume,
                output_path,
                dcm_patient,
                dcm_modality,
                dcm_study_desc,
                dcm_series_desc,
                ensure_spacing(pixel_spacing or volume.spacing),
                prefix,
            )
        return
    raise ValueError(f"Unsupported target format: {fmt}")


def canonical_format(value: str) -> str:
    if not value:
        raise ValueError("Format name is empty")
    lower = value.lower()
    return FORMAT_ALIASES.get(lower, lower)


def parse_spacing_value(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not value:
        return None
    parts = re.split(r"[ ,x]+", value.strip())
    nums = [float(p) for p in parts if p]
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(nums[-1])
    return tuple(nums[:3])


def interactive_prompt() -> Tuple[str, str, str, str, dict]:
    print("== Interactive Image Converter ==")
    fmt_map = {fmt: fmt for fmt in SUPPORTED_FORMATS}
    fmt_map.update({alias: canonical_format(alias) for alias in FORMAT_ALIASES})
    def pick_role(role: str) -> str:
        while True:
            choice = input(f"请选择{role}格式[{', '.join(SUPPORTED_FORMATS)}]: ").strip()
            try:
                return canonical_format(choice)
            except Exception:
                print("格式不支持，请重试。")
    src_fmt = pick_role("源")
    dst_fmt = pick_role("目标")
    while True:
        hint = "文件路径" if src_fmt == "nii" else "文件夹路径"
        src = input(f"输入源{hint}: ").strip()
        if src and os.path.exists(src):
            if src_fmt == "nii" and os.path.isfile(src):
                break
            if src_fmt != "nii" and os.path.isdir(src):
                break
        print("路径不符合要求，请重试。")
    dst_hint = "文件路径" if dst_fmt in ("nii", "npz") else "文件夹路径"
    dst = input(f"输出{dst_hint}：").strip()
    while not dst:
        dst = input("输出路径不能为空，请填写: ").strip()
    extra: dict = {}
    axis_input = input("切片轴 (0/1/2，默认 2): ").strip()
    extra["slice_axis"] = int(axis_input) if axis_input.isdigit() else 2
    spacing_input = input("像素/切片间距 (mm，例如 0.5,0.5,1.0): ").strip()
    parsed = parse_spacing_value(spacing_input)
    if parsed:
        extra["pixel_spacing"] = parsed
    ref = input("参考 NIfTI (可选，用于保留 affine): ").strip()
    if ref:
        extra["reference_nii"] = ref
    extra["png_prefix"] = input("PNG 前缀 (默认 slice_): ").strip() or "slice_"
    extra["png_ext"] = input("PNG 扩展名 (默认 png): ").strip() or "png"
    extra["dcm_patient"] = input("DICOM PatientName (默认 Anonymous): ").strip() or "Anonymous"
    extra["dcm_modality"] = input("DICOM Modality (CT/MR/OT) 默认 OT: ").strip() or "OT"
    extra["dcm_series_desc"] = input("DICOM SeriesDescription (默认 Converted): ").strip() or "Converted"
    extra["dcm_study_desc"] = input("DICOM StudyDescription (默认 Converted Study): ").strip() or "Converted Study"
    return src, dst, src_fmt, dst_fmt, extra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert between nii/npz/png/dcm formats")
    parser.add_argument("input", nargs="?", help="Input path (nii is file; others are folders)")
    parser.add_argument("output", nargs="?", help="Output path (nii/npz can be file; others are folders)")
    parser.add_argument("source_format", nargs="?", choices=SUPPORTED_FORMATS + tuple(FORMAT_ALIASES.keys()), help="Source format (nii/npz/png/dcm)")
    parser.add_argument("target_format", nargs="?", choices=SUPPORTED_FORMATS + tuple(FORMAT_ALIASES.keys()), help="Target format (nii/npz/png/dcm)")
    parser.add_argument("--slice-axis", type=int, default=2, help="Axis for PNG slices (0/1/2)")
    parser.add_argument("--pixel-spacing", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Voxel spacing in mm")
    parser.add_argument("--reference-nii", type=str, help="Path to NIfTI to reuse affine")
    parser.add_argument("--png-prefix", default="slice_", help="PNG filename prefix")
    parser.add_argument("--png-ext", default="png", help="PNG output extension")
    parser.add_argument("--dcm-patient", default="Anonymous", help="PatientName for exported DICOM slices")
    parser.add_argument("--dcm-modality", default="OT", help="Modality for DICOM series")
    parser.add_argument("--dcm-series-desc", default="Converted", help="SeriesDescription for DICOM series")
    parser.add_argument("--dcm-study-desc", default="Converted Study", help="StudyDescription for DICOM series")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    interactive = args.interactive or not (args.input and args.output and args.source_format and args.target_format)
    if interactive:
        src_path, dst_path, src_fmt, dst_fmt, extra = interactive_prompt()
    else:
        src_path = args.input
        dst_path = args.output
        src_fmt = canonical_format(args.source_format)
        dst_fmt = canonical_format(args.target_format)
        extra = {
            "slice_axis": args.slice_axis,
            "pixel_spacing": tuple(args.pixel_spacing) if args.pixel_spacing else None,
            "reference_nii": args.reference_nii,
            "png_prefix": args.png_prefix,
            "png_ext": args.png_ext,
            "dcm_patient": args.dcm_patient,
            "dcm_modality": args.dcm_modality,
            "dcm_series_desc": args.dcm_series_desc,
            "dcm_study_desc": args.dcm_study_desc,
        }
    try:
        if src_fmt == "nii" and not os.path.isfile(src_path):
            raise ValueError("源格式为 nii 时，输入必须是文件路径")
        if src_fmt != "nii" and not os.path.isdir(src_path):
            raise ValueError("源格式不是 nii 时，输入必须是文件夹路径")
        volumes = read_volumes(
            src_path,
            src_fmt,
            dst_fmt,
            extra.get("slice_axis", 2),
            extra.get("pixel_spacing"),
            extra.get("reference_nii"),
        )
        write_volumes(
            volumes,
            dst_path,
            dst_fmt,
            extra.get("slice_axis", 2),
            extra.get("pixel_spacing"),
            extra.get("png_prefix", "slice_"),
            extra.get("png_ext", "png"),
            extra.get("dcm_patient", "Anonymous"),
            extra.get("dcm_modality", "OT"),
            extra.get("dcm_series_desc", "Converted"),
            extra.get("dcm_study_desc", "Converted Study"),
        )
    except Exception as exc:
        print(f"转换失败: {exc}")
        return 2
    if args.verbose:
        total = len(volumes)
        src_name = os.path.basename(src_path)
        dst_name = os.path.basename(dst_path)
        print(f"{src_fmt.upper()}({src_name}) -> {dst_fmt.upper()}({dst_name}), 共 {total} 项")
    else:
        print("转换完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
