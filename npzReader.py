#!/usr/bin/env python3
"""
nzp 阅读器
功能：
1) 通过命令行参数或交互选择一个 nzp 文件
2) 将 nzp 中的所有可识别信息导出为一个 HTML 文件（包含摘要、哈希、magic bytes、文本内容、hex dump、可识别的嵌入文件列表与图像预览等）
建议安装numpy pillow以支持图片渲染

用法:
    python dcmReader.py path/to/file.nzp -o output.html --open

"""

import argparse
import base64
import hashlib
import html
import io
import json
import os
import shutil
import sys
import tarfile
import zipfile
import webbrowser
from datetime import datetime
from typing import List, Tuple
import ast
import struct

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except Exception:
    np = None
    _NUMPY_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    Image = None
    _PIL_AVAILABLE = False

MAX_TEXT_DISPLAY = 100_000


def read_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def hashes(data: bytes) -> dict:
    return {
        'md5': hashlib.md5(data).hexdigest(),
        'sha1': hashlib.sha1(data).hexdigest(),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def detect_magic(data: bytes) -> str:
    sig = data[:16]
    # common signatures
    if sig.startswith(b'PK\x03\x04'):
        return 'zip'
    if sig.startswith(b"\x1f\x8b\x08"):
        return 'gzip'
    if sig.startswith(b'%PDF'):
        return 'pdf'
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return 'png'
    if sig.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if sig.startswith(b'\x93NUMPY'):
        return 'npy'
    if sig.strip().startswith(b'{') or sig.strip().startswith(b'['):
        return 'json/text'
    return 'unknown'


def parse_npy_header(raw: bytes) -> dict:
    """Parse minimal header from .npy bytes without importing numpy."""
    try:
        if not raw.startswith(b'\x93NUMPY'):
            raise ValueError("Not an NPY file")
        major = raw[6]
        minor = raw[7]
        if major == 1:
            header_len = int.from_bytes(raw[8:10], 'little')
            header_start = 10
        else:
            header_len = int.from_bytes(raw[8:12], 'little')
            header_start = 12
        header_bytes = raw[header_start:header_start+header_len]
        header = header_bytes.decode('latin-1').strip()
        # header is Python dict literal
        header_dict = ast.literal_eval(header)
        return {
            'descr': header_dict.get('descr'),
            'fortran_order': header_dict.get('fortran_order'),
            'shape': header_dict.get('shape'),
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_npy(raw: bytes) -> dict:
    """Analyze .npy bytes: return shape, dtype, summary and optional image data_uri."""
    info = {'magic': 'npy'}
    info.update(parse_npy_header(raw))
    # try to load full array if numpy available
    if _NUMPY_AVAILABLE:
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            info['shape'] = getattr(arr, 'shape', info.get('shape'))
            info['dtype'] = str(getattr(arr, 'dtype', info.get('descr')))
            try:
                # compute summary stats for numeric arrays
                if np.issubdtype(arr.dtype, np.number):
                    info['min'] = float(np.nanmin(arr))
                    info['max'] = float(np.nanmax(arr))
                    info['mean'] = float(np.nanmean(arr))
                    info['summary'] = f"min={info['min']}, max={info['max']}, mean={info['mean']:.3f}"
            except Exception:
                pass
            # image preview for 2D grayscale or 3-channel arrays
            if _PIL_AVAILABLE and isinstance(info.get('shape'), tuple):
                if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1,3,4)):
                    try:
                        img = arr
                        # convert to uint8
                        if img.dtype != np.uint8:
                            a_min = float(np.nanmin(img))
                            a_max = float(np.nanmax(img))
                            if a_max > a_min:
                                img = (255 * (img - a_min) / (a_max - a_min)).astype(np.uint8)
                            else:
                                img = (img*0).astype(np.uint8)
                        im = Image.fromarray(img)
                        b = io.BytesIO()
                        im.save(b, format='PNG')
                        info['data_uri'] = bytes_to_data_uri(b.getvalue(), 'image/png')
                    except Exception:
                        pass
            return info
        except Exception as e:
            info['error'] = f'numpy load failed: {e}'
    return info


# extract_printable_strings removed to reduce noise and improve performance
# (Kept text preview functionality via try_text_preview.)


# hexdump removed (not needed for this workflow)


def try_text_preview(data: bytes, max_chars=MAX_TEXT_DISPLAY) -> Tuple[str, str]:
    """尝试将数据作为文本显示，返回 (decoded_text_or_empty, encoding_used)"""
    # try utf-8
    for enc in ('utf-8', 'latin-1', 'utf-16'):
        try:
            txt = data.decode(enc)
            preview = txt[:max_chars]
            return preview, enc
        except Exception:
            continue
    return '', ''


def is_json_like(s: str) -> bool:
    s_strip = s.lstrip()
    return s_strip.startswith('{') or s_strip.startswith('[')


def pretty_json(s: str) -> str:
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return s


def bytes_to_data_uri(b: bytes, mime: str) -> str:
    b64 = base64.b64encode(b).decode('ascii')
    return f'data:{mime};base64,{b64}'


def analyze_zip(data: bytes) -> List[dict]:
    results = []
    bio = io.BytesIO(data)
    try:
        with zipfile.ZipFile(bio) as z:
            for info in z.infolist():
                entry = {
                    'name': info.filename,
                    'size': info.file_size,
                    'compress_size': info.compress_size,
                }
                try:
                    with z.open(info) as ef:
                        raw = ef.read()
                        entry['magic'] = detect_magic(raw)
                        # handle numpy arrays specially
                        if entry['name'].lower().endswith('.npy') or entry['magic'] == 'npy':
                            try:
                                npinfo = analyze_npy(raw)
                                entry.update(npinfo)
                                # small arrays -> show small text preview
                                if 'summary' in npinfo:
                                    entry['text_preview'] = npinfo.get('summary')
                            except Exception as e:
                                entry['error'] = str(e)
                        else:
                            # small text files -> preview
                            text_preview, enc = try_text_preview(raw, max_chars=2000)
                            if text_preview:
                                entry['text_preview'] = text_preview
                                entry['encoding'] = enc
                            # images inline
                            if entry['magic'] in ('png', 'jpeg'):
                                mime = 'image/png' if entry['magic']=='png' else 'image/jpeg'
                                entry['data_uri'] = bytes_to_data_uri(raw, mime)
                except Exception as e:
                    entry['error'] = str(e)
                results.append(entry)
    except zipfile.BadZipFile:
        return []
    return results


def analyze_tar(data: bytes) -> List[dict]:
    results = []
    bio = io.BytesIO(data)
    try:
        with tarfile.open(fileobj=bio, mode='r:*') as t:
            for member in t.getmembers():
                entry = {
                    'name': member.name,
                    'size': member.size,
                }
                try:
                    if member.isfile():
                        f = t.extractfile(member)
                        if f:
                            raw = f.read()
                            entry['magic'] = detect_magic(raw)
                            text_preview, enc = try_text_preview(raw, max_chars=2000)
                            if text_preview:
                                entry['text_preview'] = text_preview
                                entry['encoding'] = enc
                            if entry.get('magic') in ('png','jpeg'):
                                mime = 'image/png' if entry['magic']=='png' else 'image/jpeg'
                                entry['data_uri'] = bytes_to_data_uri(raw, mime)
                except Exception as e:
                    entry['error'] = str(e)
                results.append(entry)
    except tarfile.ReadError:
        return []
    return results


def generate_html_report(path: str, data: bytes, out_path: str) -> None:
    meta = hashes(data)
    magic = detect_magic(data)
    # removed extraction of printable strings and hex dump to reduce noise
    text_preview, enc = try_text_preview(data)

    zip_entries = analyze_zip(data) if magic == 'zip' else []
    tar_entries = analyze_tar(data) if magic in ('gzip','unknown') else []

    now = datetime.utcnow().isoformat() + 'Z'

    # escape helper
    def esc(s: str) -> str:
        return html.escape(s)

    # start HTML
    html_parts = []
    html_parts.append('<!doctype html>')
    html_parts.append('<html><head><meta charset="utf-8"><title>NZP Report</title>')
    html_parts.append('<style>body{font-family:Inter, -apple-system, system-ui, Roboto, "Helvetica Neue", Arial, sans-serif; padding:20px; line-height:1.5} h1,h2{color:#111} pre{background:#f8f8f8;padding:12px;border-radius:6px;overflow:auto} table{border-collapse:collapse} td,th{padding:6px;border:1px solid #ddd;text-align:left} .m{font-size:0.9em;color:#555}</style>')
    html_parts.append('</head><body>')
    html_parts.append(f'<h1>NZP 报告 — {esc(os.path.basename(path))}</h1>')
    html_parts.append(f'<p class="m">生成时间 (UTC): {now}</p>')

    # Summary
    html_parts.append('<h2>摘要 ✅</h2>')
    html_parts.append('<table>')
    html_parts.append(f'<tr><th>文件</th><td>{esc(path)}</td></tr>')
    html_parts.append(f'<tr><th>大小</th><td>{len(data):,} bytes</td></tr>')
    html_parts.append(f'<tr><th>检测类型</th><td>{esc(magic)}</td></tr>')
    html_parts.append(f'<tr><th>MD5</th><td>{meta["md5"]}</td></tr>')
    html_parts.append(f'<tr><th>SHA1</th><td>{meta["sha1"]}</td></tr>')
    html_parts.append(f'<tr><th>SHA256</th><td>{meta["sha256"]}</td></tr>')
    html_parts.append('</table>')

    # Text preview
    html_parts.append('<h2>文本预览 ✍️</h2>')
    if text_preview:
        # if json-like then pretty
        if is_json_like(text_preview):
            pretty = pretty_json(text_preview)
            html_parts.append('<pre>' + esc(pretty) + '</pre>')
        else:
            html_parts.append('<pre>' + esc(text_preview[:MAX_TEXT_DISPLAY]) + '</pre>')
            if len(text_preview) > MAX_TEXT_DISPLAY:
                html_parts.append('<p class="m">(文本被截断，太大了)</p>')
        if enc:
            html_parts.append(f'<p class="m">使用编码: {esc(enc)}</p>')
    else:
        html_parts.append('<p class="m">未检测到可直接作为文本显示的内容。</p>')

    # Removed '可打印字符串' and 'Hex Dump' sections to declutter report
    html_parts.append('<p class="m">提示: 已移除 Hex Dump 与可打印字符串以减少噪音。</p>')

    # Zip contents
    if zip_entries:
        html_parts.append('<h2>ZIP 嵌套文件 📦</h2>')
        html_parts.append('<table>')
        html_parts.append('<tr><th>名称</th><th>大小</th><th>压缩大小</th><th>类型</th><th>预览</th></tr>')
        for e in zip_entries:
            preview = ''
            meta = []
            if e.get('magic') == 'npy' or e['name'].lower().endswith('.npy'):
                if 'shape' in e:
                    meta.append(f"shape: {e.get('shape')}")
                dtype = e.get('dtype') or e.get('descr')
                if dtype:
                    meta.append(f"dtype: {esc(str(dtype))}")
                if 'fortran_order' in e:
                    meta.append(f"fortran_order: {e.get('fortran_order')}")
                stats = []
                for k in ('min', 'max', 'mean'):
                    if k in e:
                        stats.append(f"{k}={e.get(k)}")
                if stats:
                    meta.append(', '.join(stats))
                if 'summary' in e:
                    meta.append(esc(str(e.get('summary'))))
            if meta:
                preview += '<div class="m">' + ' | '.join(meta) + '</div>'
            if 'text_preview' in e:
                preview += '<pre>' + esc(e['text_preview'][:2000]) + '</pre>'
            if 'data_uri' in e:
                preview += f'<img src="{e["data_uri"]}" style="max-width:400px;max-height:300px;">'
            if 'error' in e:
                preview += f'<div class="m">Error: {esc(e["error"])}</div>'
            html_parts.append(f"<tr><td>{esc(e['name'])}</td><td>{e.get('size','')}</td><td>{e.get('compress_size','')}</td><td>{esc(e.get('magic',''))}</td><td>{preview}</td></tr>")
        html_parts.append('</table>')

    # Tar contents
    if tar_entries:
        html_parts.append('<h2>TAR 嵌套文件 🗂️</h2>')
        html_parts.append('<table>')
        html_parts.append('<tr><th>名称</th><th>大小</th><th>类型</th><th>预览</th></tr>')
        for e in tar_entries:
            preview = ''
            if 'text_preview' in e:
                preview = '<pre>' + esc(e['text_preview'][:2000]) + '</pre>'
            if 'data_uri' in e:
                preview = f'<img src="{e["data_uri"]}" style="max-width:400px;max-height:300px;">'
            html_parts.append(f"<tr><td>{esc(e['name'])}</td><td>{e.get('size','')}</td><td>{esc(e.get('magic',''))}</td><td>{preview}</td></tr>")
        html_parts.append('</table>')

    if not _NUMPY_AVAILABLE:
        html_parts.append('<p class="m">提示: 未检测到 numpy，无法计算数组统计或生成可视化预览。安装 numpy 可获得更好结果。</p>')
    if not _PIL_AVAILABLE:
        html_parts.append('<p class="m">提示: 未检测到 Pillow (PIL)，无法生成图像预览。安装 Pillow 可生成 PNG 预览。</p>')
    html_parts.append('<hr>')
    html_parts.append('<p class="m">报告由 nzp 阅读器生成</p>')
    html_parts.append('</body></html>')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))


def main():
    parser = argparse.ArgumentParser(description='NZP 阅读器 — 导出 NZP 到 HTML 报告')
    parser.add_argument('path', nargs='?', help='NZP 文件路径')
    parser.add_argument('-o', '--output', help='输出 HTML 文件路径 (默认为 <input>.nzp.html)')
    parser.add_argument('--open', action='store_true', help='生成后自动在默认浏览器中打开')
    args = parser.parse_args()

    path = args.path
    if not path:
        try:
            path = input('请输入 NZP 文件路径: ').strip()
        except EOFError:
            print('没有提供文件路径，退出。')
            sys.exit(1)

    if not os.path.isfile(path):
        print(f'文件不存在: {path}')
        sys.exit(1)

    data = read_bytes(path)

    out_path = args.output if args.output else path + '.html'

    print('正在分析，可能需要一些时间...')
    try:
        generate_html_report(path, data, out_path)
        print(f'HTML 报告已生成: {out_path}')
        if args.open:
            webbrowser.open('file://' + os.path.abspath(out_path))
    except Exception as e:
        print(f'生成报告时出错: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
