from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
from pathlib import Path
import struct
import zlib

import numpy as np

NPZ_EMBED_MAGIC = b"NPZ_ROUNDTRIP_V1\0"
NPZ_LEN_FMT = "<Q"
DCM_PRIVATE_CREATOR_TAG = (0x0011, 0x0010)
DCM_PAYLOAD_TAG = (0x0011, 0x1010)
NIFTI_EXT_CODE = 40
PNG_IMAGE_EMBED_KEY = "NPZ_IMAGE_NPY_B64Z"

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
except ImportError:
    pydicom = None
    Dataset = None
    FileDataset = None
    FileMetaDataset = None
    ExplicitVRLittleEndian = None
    SecondaryCaptureImageStorage = None
    generate_uid = None


def _require_nibabel() -> None:
    if nib is None:
        raise RuntimeError("需要 nibabel，请先安装：pip install nibabel")


def _require_pydicom() -> None:
    if (
        pydicom is None
        or Dataset is None
        or FileDataset is None
        or FileMetaDataset is None
        or ExplicitVRLittleEndian is None
        or SecondaryCaptureImageStorage is None
        or generate_uid is None
    ):
        raise RuntimeError("需要 pydicom，请先安装：pip install pydicom")


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("需要 Pillow，请先安装：pip install pillow")


def _normalize_png_name(name: str) -> list[str]:
    parts = []
    current = ""
    digit_mode = False
    for ch in name:
        if ch.isdigit() and not digit_mode:
            if current:
                parts.append(current.lower())
            current = ch
            digit_mode = True
        elif ch.isdigit() and digit_mode:
            current += ch
        elif not ch.isdigit() and digit_mode:
            parts.append(int(current))
            current = ch
            digit_mode = False
        else:
            current += ch
    if current:
        parts.append(int(current) if digit_mode else current.lower())
    return parts


def _to_label_dtype(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    return np.rint(arr).astype(np.int16)


def save_onnx_compatible_npz(
    out_path: Path,
    image: np.ndarray,
    label: np.ndarray | None,
) -> None:
    image_arr = np.asarray(image)
    if image_arr.ndim != 2:
        raise ValueError(f"image 需要是二维数组，当前形状：{image_arr.shape}")

    if label is None:
        label_arr = np.zeros(image_arr.shape, dtype=np.uint8)
    else:
        label_arr = np.asarray(label)
        if label_arr.shape != image_arr.shape:
            raise ValueError(
                f"label 形状需与 image 一致，当前 image={image_arr.shape}, label={label_arr.shape}"
            )
        label_arr = _to_label_dtype(label_arr)

    np.savez(out_path, image=image_arr, label=label_arr)


def _read_file_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _write_file_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)


def _pack_embedded_npz(npz_bytes: bytes) -> bytes:
    encoded = base64.b64encode(npz_bytes)
    return NPZ_EMBED_MAGIC + struct.pack(NPZ_LEN_FMT, len(encoded)) + encoded


def _unpack_embedded_npz(payload: bytes) -> bytes | None:
    if payload.startswith(NPZ_EMBED_MAGIC):
        start = len(NPZ_EMBED_MAGIC)
        end = start + struct.calcsize(NPZ_LEN_FMT)
        if len(payload) < end:
            return None
        expected_len = struct.unpack(NPZ_LEN_FMT, payload[start:end])[0]
        encoded = payload[end : end + expected_len]
        if len(encoded) != expected_len:
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception:
            return None
    return None


def _extension_to_bytes(ext: object) -> bytes:
    raw = getattr(ext, "_raw", b"")
    if raw:
        return bytes(raw)
    try:
        content = ext.get_content()  # type: ignore[attr-defined]
    except Exception:
        return b""
    if isinstance(content, str):
        return content.encode("utf-8", errors="ignore")
    return bytes(content)


def dcm_to_npz(input_path: Path, out_path: Path) -> None:
    _require_pydicom()
    ds = pydicom.dcmread(str(input_path), force=True)

    if DCM_PAYLOAD_TAG in ds:
        raw_payload = bytes(ds[DCM_PAYLOAD_TAG].value)
        embedded = _unpack_embedded_npz(raw_payload)
        if embedded is not None:
            _write_file_bytes(out_path, embedded)
            return

    image = ds.pixel_array
    save_onnx_compatible_npz(out_path, image=image, label=None)


def nii_to_npz(input_path: Path, out_path: Path, slice_index: int | None) -> None:
    _require_nibabel()
    nii = nib.load(str(input_path))

    for ext in nii.header.extensions:
        embedded = _unpack_embedded_npz(_extension_to_bytes(ext))
        if embedded is not None:
            _write_file_bytes(out_path, embedded)
            return

    data = np.asarray(nii.get_fdata())
    if data.ndim == 2:
        image = data
    elif data.ndim == 3:
        if slice_index is None:
            slice_index = data.shape[2] // 2
        if slice_index < 0 or slice_index >= data.shape[2]:
            raise ValueError(f"slice_index 越界：{slice_index}，合法范围 [0, {data.shape[2] - 1}]")
        image = data[:, :, slice_index]
    else:
        raise ValueError(f"暂不支持 4D 及以上 NIfTI，当前形状：{data.shape}")

    save_onnx_compatible_npz(out_path, image=image, label=None)


def _read_png_gray(path: Path) -> np.ndarray:
    _require_pillow()
    with Image.open(path) as img:
        embedded = img.info.get(PNG_IMAGE_EMBED_KEY)
        if embedded:
            try:
                payload = base64.b64decode(str(embedded))
                npy_bytes = zlib.decompress(payload)
                arr = np.load(io.BytesIO(npy_bytes), allow_pickle=False)
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    return np.asarray(arr)
            except Exception:
                pass
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        return gray / 255.0


def png_to_npz(input_path: Path, out_path: Path) -> None:
    if input_path.is_file():
        image = _read_png_gray(input_path)
        save_onnx_compatible_npz(out_path, image=image, label=None)
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"未找到 png 文件或目录：{input_path}")

    png_files = sorted(
        [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"],
        key=lambda p: _normalize_png_name(p.name),
    )
    if not png_files:
        raise ValueError(f"目录下没有 png 文件：{input_path}")

    stack = np.stack([_read_png_gray(p) for p in png_files], axis=0)
    image = stack[len(stack) // 2]
    save_onnx_compatible_npz(out_path, image=image, label=None)


def _load_npz_image(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as archive:
        if key not in archive.files:
            raise KeyError(f"npz 中找不到键 '{key}'，可用键：{archive.files}")
        data = np.asarray(archive[key])
    return data


def _normalize_2d(arr: np.ndarray, clip: bool) -> np.ndarray:
    work = np.asarray(arr)
    if work.ndim == 3:
        work = work[:, :, work.shape[2] // 2]
    if work.ndim != 2:
        raise ValueError(f"仅支持 2D 或 3D 数据导出，当前形状：{work.shape}")

    if np.issubdtype(work.dtype, np.integer):
        return work

    if clip:
        work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
        work = np.clip(work, 0.0, 65535.0)
        return np.rint(work).astype(np.uint16)

    return work.astype(np.float32)


def npz_to_dcm(input_path: Path, out_path: Path, key: str) -> None:
    _require_pydicom()
    image = _normalize_2d(_load_npz_image(input_path, key=key), clip=True)
    packed_npz = _pack_embedded_npz(_read_file_bytes(input_path))

    now = dt.datetime.now()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.PatientName = "Converted^FromNPZ"
    ds.PatientID = "NPZ0001"
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.InstanceNumber = 1
    ds.PixelData = np.asarray(image, dtype=np.uint16).tobytes()
    ds.add_new(DCM_PRIVATE_CREATOR_TAG, "LO", "NPZ_ROUNDTRIP")
    ds.add_new(DCM_PAYLOAD_TAG, "OB", packed_npz)

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(out_path), write_like_original=False)


def npz_to_nii(input_path: Path, out_path: Path, key: str) -> None:
    _require_nibabel()
    image = _load_npz_image(input_path, key=key)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"仅支持 2D/3D 写入 NIfTI，当前形状：{image.shape}")
    nii = nib.Nifti1Image(image.astype(np.float32), affine=np.eye(4, dtype=np.float32))
    packed_npz = _pack_embedded_npz(_read_file_bytes(input_path))
    nii.header.extensions.append(nib.nifti1.Nifti1Extension(NIFTI_EXT_CODE, packed_npz))
    nib.save(nii, str(out_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="医疗影像格式转换：dcm/nii/png 与 npz")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dcm2npz", "nii2npz", "png2npz", "npz2dcm", "npz2nii"],
        help="转换模式",
    )
    parser.add_argument("--input", required=True, help="输入路径")
    parser.add_argument("--output", required=True, help="输出路径")
    parser.add_argument(
        "--npz-key",
        default="image",
        help="npz->dcm/nii 时读取的键名（默认 image）",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="nii2npz 时 3D 体数据选择的切片索引，默认取中间切片",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dcm2npz":
        dcm_to_npz(input_path, output_path)
    elif args.mode == "nii2npz":
        nii_to_npz(input_path, output_path, slice_index=args.slice_index)
    elif args.mode == "png2npz":
        png_to_npz(input_path, output_path)
    elif args.mode == "npz2dcm":
        npz_to_dcm(input_path, output_path, key=args.npz_key)
    elif args.mode == "npz2nii":
        npz_to_nii(input_path, output_path, key=args.npz_key)
    else:
        raise ValueError(f"不支持的 mode: {args.mode}")

    print(f"转换完成: {args.mode} -> {output_path}")


if __name__ == "__main__":
    main()
