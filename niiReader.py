#!/usr/bin/env python3
"""
NIfTI 阅读器
功能：
1. 通过命令行参数或交互选择一个 .nii 或 .nii.gz 文件
2. 将 NIfTI 文件中的所有信息导出为 HTML 文件（用户选择保存位置，默认为程序根目录）

依赖：
- nibabel (pip install nibabel)
- numpy
- tkinter (通常内置)

用法：
    python niiReader.py path/to/file.nii -o output.html
    python niiReader.py  # 交互式选择文件
"""

import argparse
import os
import sys
import json
import base64
import io
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    _NIBABEL_AVAILABLE = True
except ImportError:
    print("错误：需要安装 nibabel, numpy, pillow。运行 'pip install nibabel numpy pillow'")
    sys.exit(1)

def load_nii(file_path):
    """加载 NIfTI 文件并返回图像对象"""
    try:
        img = nib.load(file_path)
        return img
    except Exception as e:
        print(f"加载文件失败：{e}")
        return None

def get_nii_info(img):
    """提取 NIfTI 文件的信息"""
    info = {}
    
    # 头部信息
    header = img.header
    info['header'] = {
        'data_type': str(header.get_data_dtype()),
        'data_shape': tuple(header.get_data_shape()),
        'zooms': tuple(float(z) for z in header.get_zooms()),
        'qform_code': int(header['qform_code'].item()) if 'qform_code' in header else None,
        'sform_code': int(header['sform_code'].item()) if 'sform_code' in header else None,
        'intent_name': header['intent_name'].item().decode('utf-8').strip() if 'intent_name' in header else '',
        'descrip': header['descrip'].item().decode('utf-8').strip() if 'descrip' in header else '',
    }
    
    # 完整头部信息
    full_header = {}
    for key in header.keys():
        try:
            value = header[key]
            if hasattr(value, 'item'):
                value = value.item()
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore').strip()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            full_header[key] = value
        except:
            full_header[key] = str(value)
    info['full_header'] = full_header
    
    # 数据信息
    data = img.get_fdata()
    info['data'] = {
        'shape': tuple(data.shape),
        'dtype': str(data.dtype),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'unique_values': int(len(np.unique(data))) if data.size < 1000000 else 'Too many to count',
    }
    
    # 仿射矩阵
    info['affine'] = [list(float(x) for x in row) for row in img.affine.tolist()]
    
    # 生成图像预览
    images = generate_image_previews(data)
    info['images'] = images
    
    return info

def generate_image_previews(data):
    """生成图像预览的base64编码"""
    images = {}
    
    # 归一化数据到0-255
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        data_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        data_norm = np.zeros_like(data, dtype=np.uint8)
    
    # 生成三个平面的中间切片
    slices = {}
    if data.shape[2] > 0:  # 轴向 (axial)
        mid = data.shape[2] // 2
        slice_data = data_norm[:, :, mid]
        slices['axial'] = slice_data
    
    if data.shape[1] > 0:  # 冠状 (coronal)
        mid = data.shape[1] // 2
        slice_data = data_norm[:, mid, :]
        slices['coronal'] = slice_data
    
    if data.shape[0] > 0:  # 矢状 (sagittal)
        mid = data.shape[0] // 2
        slice_data = data_norm[mid, :, :]
        slices['sagittal'] = slice_data
    
    # 生成base64图像
    for plane, slice_data in slices.items():
        img = Image.fromarray(slice_data, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images[plane] = f"data:image/png;base64,{img_base64}"
    
    return images

def generate_html(info, file_path):
    """生成 HTML 内容"""
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIfTI 文件信息 - {os.path.basename(file_path)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>NIfTI 文件信息</h1>
    <p><strong>文件路径：</strong> {file_path}</p>
    
    <h2>头部信息</h2>
    <table>
        <tr><th>属性</th><th>值</th></tr>
"""
    
    for key, value in info['header'].items():
        html += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html += "    </table>\n"
    
    html += """
    <h2>数据信息</h2>
    <table>
        <tr><th>属性</th><th>值</th></tr>
"""
    
    for key, value in info['data'].items():
        html += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html += "    </table>\n"
    
    html += f"""
    <h2>仿射矩阵</h2>
    <pre>{json.dumps(info['affine'], indent=2)}</pre>
    
    <h2>图像预览</h2>
"""
    
    if info['images']:
        for plane, img_src in info['images'].items():
            html += f"""
    <h3>{plane.capitalize()} 平面</h3>
    <img src="{img_src}" alt="{plane} slice" style="max-width: 100%; height: auto; border: 1px solid #ddd;">
"""
    else:
        html += "<p>无图像数据</p>\n"
    
    html += f"""
    <h2>完整头部信息</h2>
    <pre>{json.dumps(info['full_header'], indent=2)}</pre>
</body>
</html>
"""
    
    return html

def main():
    parser = argparse.ArgumentParser(description="NIfTI 文件阅读器")
    parser.add_argument('file', nargs='?', help='NIfTI 文件路径 (.nii 或 .nii.gz)')
    parser.add_argument('-o', '--output', help='输出 HTML 文件路径')
    
    args = parser.parse_args()
    
    # 获取文件路径
    if args.file:
        file_path = args.file
    else:
        # 交互式输入文件路径
        file_path = input("请输入 NIfTI 文件路径 (.nii 或 .nii.gz): ").strip()
        if not file_path:
            print("未输入文件路径，退出。")
            return
    
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return
    
    # 加载 NIfTI 文件
    img = load_nii(file_path)
    if img is None:
        return
    
    # 提取信息
    info = get_nii_info(img)
    
    # 获取输出路径
    if args.output:
        output_path = args.output
    else:
        # 默认保存到程序根目录
        script_dir = Path(__file__).parent
        default_name = Path(file_path).stem + "_info.html"
        output_path = str(script_dir / default_name)
    
    # 生成 HTML
    html_content = generate_html(info, file_path)
    
    # 保存 HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 文件已保存到：{output_path}")

if __name__ == "__main__":
    main()