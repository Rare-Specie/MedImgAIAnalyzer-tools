#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-One 医学影像小工具（单文件版）

这个脚本用于把一组 2D `.npz` 切片转换为：
1) 2D 浏览 HTML（带切片滑动与标注开关）
2) 3D 网格 `.glb`
3) 内嵌 GLB 的 3D HTML

同时也支持把多个现有 `.glb` 文件批量打包成可直接打开的 HTML。

特点：
- 单文件运行，不依赖仓库内其他 Python 文件
- 启动随机展示中文冷笑话（内置 100 条）
- 交互模式 + 命令行参数模式
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import struct
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

try:
    from skimage.measure import marching_cubes
except Exception:
    marching_cubes = None


RAW_KEYS = ["image", "img", "raw", "ct", "data", "slice", "input"]
ANN_KEYS = ["label", "mask", "seg", "annotation", "gt"]


COLD_JOKES = [
    "1. 为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 == Dec 25。",
    "2. 有人问我 bug 在哪，我说在电脑里，不在我心里。",
    "3. 医生说我缺钙，我说那我多看点代码，反正都是骨架。",
    "4. 我减肥失败了，因为每次都先把计划写成了 TODO。",
    "5. 为什么键盘很累？因为它总被敲。",
    "6. 我把闹钟名字改成“起床暴富”，结果它每天都在骗人。",
    "7. Wi-Fi 一断，我就知道我和世界只是“局域网关系”。",
    "8. 我不是拖延，我只是把未来安排得很满。",
    "9. 为什么文件总丢？因为它觉得自己应该“云游四海”。",
    "10. 我问 AI 会不会取代我，AI 说：先把需求写清楚。",
    "11. 为什么杯子会生气？因为总有人给它“添堵”。",
    "12. 我最擅长早睡——在白天。",
    "13. 老板说要结果导向，我就把显示器转向了他。",
    "14. 你知道什么叫稳定发挥吗？每次都卡在同一行。",
    "15. 为什么猫喜欢键盘？因为有很多“喵键位”。",
    "16. 我不怕困难，我怕困难还要开会。",
    "17. 失败不可怕，可怕的是失败还要写复盘。",
    "18. 为什么路由器最懂爱？因为它会“连你”。",
    "19. 我买了智能体重秤，它很智能，从不夸我。",
    "20. 为什么雨伞总被借走？因为它“伞良”。",
    "21. 我的人生目标很明确：先吃饭，再考虑世界和平。",
    "22. 为什么电梯最会安慰人？它总说“别急，马上到”。",
    "23. 我一打开文档，灵感就自动最小化。",
    "24. 为什么硬盘记性好？因为它从不“失忆”。",
    "25. 我不是选择困难，我是每个都想试试。",
    "26. 为什么闹钟不受欢迎？因为它总揭穿美梦。",
    "27. 今天状态很好，已经成功把昨天的问题带到了今天。",
    "28. 为什么耳机总打结？因为它们关系太复杂。",
    "29. 我在学时间管理，先从管理午睡开始。",
    "30. 为什么手机电量低会焦虑？因为它“电”不到你。",
    "31. 人生就像进度条，看着在动，其实在缓冲。",
    "32. 为什么月亮不加班？因为它只上夜班。",
    "33. 我本来想早起，后来决定给太阳一点面子。",
    "34. 为什么打印机脾气大？因为它总“卡纸气”。",
    "35. 我最会的运动：左右横跳于需求之间。",
    "36. 为什么冰箱最会保密？因为它“冷处理”。",
    "37. 计划赶不上变化，变化赶不上我先躺下。",
    "38. 为什么铅笔很自信？因为它有“芯”。",
    "39. 我不是没灵感，我是在等灵感排队叫号。",
    "40. 为什么电风扇很讲道理？因为它一直在“转述”。",
    "41. 我喝咖啡不是为了清醒，是为了看起来在努力。",
    "42. 为什么楼梯不怕累？因为它天生有“台阶”。",
    "43. 今天不想努力，明天再努力劝今天的我。",
    "44. 为什么钟表很守时？因为它不敢停。",
    "45. 我喜欢周一，因为离周末最近的是上周末。",
    "46. 为什么橡皮擦很低调？因为它专门“抹去存在感”。",
    "47. 我的作息很规律：困了就困。",
    "48. 为什么书包很能装？因为它有“容量担当”。",
    "49. 我不是社恐，我是社交省电模式。",
    "50. 为什么门把手最热情？因为它总主动“握手”。",
    "51. 我把目标定得很远，这样就不会轻易到达。",
    "52. 为什么牙刷很勤奋？因为它每天都“刷存在”。",
    "53. 今天心情像云：有点散，但不下班。",
    "54. 为什么枕头最懂你？因为它接住你的所有想法。",
    "55. 我在学习理财，先从不乱买第二杯奶茶开始。",
    "56. 为什么镜子不会说谎？因为它只会反射，不会反驳。",
    "57. 我问天气为什么变冷，它说给你降降火。",
    "58. 为什么笔记本电脑怕摔？因为它“本子”薄。",
    "59. 我不熬夜，我只是和凌晨有合作项目。",
    "60. 为什么云盘容量总不够？因为回忆太重。",
    "61. 我今天很高效，一次性打开了十个标签页。",
    "62. 为什么窗帘爱摸鱼？因为它只负责“拉开局面”。",
    "63. 我不是路痴，我是探索型导航。",
    "64. 为什么茶杯总烫手？因为它很“热情”。",
    "65. 我的人生建议：先吃饭，后输入。",
    "66. 为什么日历容易瘦？因为它天天“掉页”。",
    "67. 我最擅长的编程语言：等下就写。",
    "68. 为什么空调很公平？因为它一视同“凉”。",
    "69. 我不怕麻烦，我怕麻烦有附件。",
    "70. 为什么楼道回声大？因为它很会“复读”。",
    "71. 我今天运动了，手指在键盘上来回冲刺。",
    "72. 为什么水果刀很直率？因为它“切中要害”。",
    "73. 我不是健忘，我是给生活留悬念。",
    "74. 为什么拖鞋很佛系？因为它总说“随便”。",
    "75. 我把烦恼写下来，发现纸比我更有压力。",
    "76. 为什么手机壳最忠诚？因为它总在外面扛伤害。",
    "77. 我不是吃货，我是食物质量监督员。",
    "78. 为什么快递盒总开心？因为它天天“拆盲盒”。",
    "79. 我的人生高光时刻：Wi-Fi 满格。",
    "80. 为什么鼠标很谦虚？因为它总说“我点到为止”。",
    "81. 我决定早睡，从明天的昨天开始。",
    "82. 为什么杯垫很重要？因为它懂得“承受”。",
    "83. 我以为我很稳，直到网突然断了。",
    "84. 为什么电池喜欢安静？因为它怕“漏电情绪”。",
    "85. 我不是在发呆，我在后台编译梦想。",
    "86. 为什么围巾很会聊天？因为它总能“接上话”。",
    "87. 我今天不内耗了，改外包给明天。",
    "88. 为什么开关很果断？因为它只有开和关。",
    "89. 我想成为更好的自己，先把闹钟音量调大。",
    "90. 为什么键帽容易丢？因为它太想“出头”。",
    "91. 我的方向感很好，永远能走到饭点。",
    "92. 为什么海绵很乐观？因为挤一挤总还有。",
    "93. 我说要断舍离，先把外卖软件静音。",
    "94. 为什么胶带很执着？因为它总想“粘住机会”。",
    "95. 我不是情绪化，我只是响应式设计。",
    "96. 为什么路灯最温柔？因为它总在夜里等你。",
    "97. 我把手机放远一点，焦虑也只远了一点。",
    "98. 为什么尺子最讲原则？因为它凡事有度。",
    "99. 我今天学会了自律：奶茶改成了大杯无糖。",
    "100. 为什么云朵不加班？因为它只负责“飘过”。",
]


@dataclass
class Mode1Settings:
    export_2d: bool = False
    out_2d: str = "output"
    export_glb: bool = False
    out_glb: str = "output"
    export_3d: bool = True
    out_3d: str = "output"
    ann_threshold: float = 0.5


@dataclass
class Mode2Settings:
    out_html: str = "output"


def natural_sort_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def detect_arrays(npz_obj) -> tuple[np.ndarray | None, np.ndarray | None]:
    files = list(npz_obj.files)
    raw = ann = None

    for key in RAW_KEYS:
        if key in files:
            raw = npz_obj[key]
            break

    for key in ANN_KEYS:
        if key in files:
            ann = npz_obj[key]
            break

    arrays = [(k, npz_obj[k]) for k in files]
    if raw is None:
        for _, value in arrays:
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.dtype != bool:
                raw = value
                break

    if ann is None:
        for _, value in arrays:
            if value is raw:
                continue
            if isinstance(value, np.ndarray) and value.ndim >= 2 and (
                value.dtype == bool or np.issubdtype(value.dtype, np.integer)
            ):
                unique_values = np.unique(value)
                if unique_values.size <= 8:
                    ann = value
                    break

    return raw, ann


def reduce_to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] <= 4:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return arr[..., 0]
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed
    raise ValueError(f"无法压缩到 2D，当前 shape={arr.shape}")


def normalize_u8(volume: np.ndarray) -> np.ndarray:
    mmin, mmax = float(np.nanmin(volume)), float(np.nanmax(volume))
    if mmax > mmin:
        return ((volume - mmin) / (mmax - mmin) * 255.0).astype(np.uint8)
    return np.clip(volume, 0, 255).astype(np.uint8)


def to_png_base64(gray2d_u8: np.ndarray) -> str:
    image = Image.fromarray(gray2d_u8.astype(np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay_base64(raw_u8: np.ndarray, ann_raw: np.ndarray, alpha: float = 0.4) -> str:
    rgb = np.stack([raw_u8] * 3, axis=-1).astype(np.float32)

    yellow_mask = ann_raw > 1.0
    red_mask = (ann_raw > 0.0) & (ann_raw <= 1.0)

    if yellow_mask.any():
        rgb[yellow_mask, 0] = rgb[yellow_mask, 0] * (1 - alpha) + 255 * alpha
        rgb[yellow_mask, 1] = rgb[yellow_mask, 1] * (1 - alpha) + 255 * alpha
        rgb[yellow_mask, 2] = rgb[yellow_mask, 2] * (1 - alpha)

    if red_mask.any():
        rgb[red_mask, 0] = rgb[red_mask, 0] * (1 - alpha) + 255 * alpha
        rgb[red_mask, 1] = rgb[red_mask, 1] * (1 - alpha)
        rgb[red_mask, 2] = rgb[red_mask, 2] * (1 - alpha)

    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def gather_paths_from_tokens(tokens: Sequence[str], ext: str) -> List[Path]:
    paths: list[Path] = []
    for token in tokens:
        p = Path(token).expanduser()
        if p.is_file() and p.suffix.lower() == ext.lower():
            paths.append(p.resolve())
    return sorted(paths, key=lambda x: natural_sort_key(str(x)))


def gather_paths_from_dir(directory: str | Path, ext: str) -> List[Path]:
    d = Path(directory).expanduser()
    if not d.exists() or not d.is_dir():
        return []
    files = [p.resolve() for p in d.rglob(f"*{ext}") if p.is_file()]
    return sorted(files, key=lambda x: natural_sort_key(str(x)))


def parse_input_to_files(raw_input: str, ext: str) -> List[Path]:
    text = raw_input.strip()
    if not text:
        return []
    as_path = Path(text).expanduser()
    if as_path.exists() and as_path.is_dir():
        return gather_paths_from_dir(as_path, ext)
    tokens = text.split()
    return gather_paths_from_tokens(tokens, ext)


def make_2d_html(payload: dict) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>NPZ 2D Viewer</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 16px; }}
    .toolbar {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; }}
    img {{ max-width: 100%; image-rendering: pixelated; border:1px solid #ddd; }}
    #info {{ margin-top:6px; color:#444; font-size:90%; }}
  </style>
</head>
<body>
  <h3>NPZ 2D Viewer</h3>
  <div class=\"toolbar\">
    <button id=\"prev\">Prev</button>
    <input id=\"zslider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" />
    <button id=\"next\">Next</button>
    <label><input id=\"showAnn\" type=\"checkbox\" checked /> 显示标注</label>
  </div>
  <img id=\"sliceImg\" src=\"\" />
  <div id=\"info\"></div>
<script>
const PAYLOAD = {json.dumps(payload, ensure_ascii=False)};
const rawImages = PAYLOAD.raw_images;
const overlayImages = PAYLOAD.overlay_images;
const zCount = PAYLOAD.z_count;
const width = PAYLOAD.width;
const height = PAYLOAD.height;

const zslider = document.getElementById('zslider');
const sliceImg = document.getElementById('sliceImg');
const showAnn = document.getElementById('showAnn');
const info = document.getElementById('info');

zslider.max = Math.max(0, zCount - 1);

function updateSlice(z) {{
  z = Math.min(Math.max(0, z | 0), zCount - 1);
  zslider.value = z;
  sliceImg.src = showAnn.checked ? overlayImages[z] : rawImages[z];
  info.innerText = `slice ${{z+1}}/${{zCount}} (${{width}}x${{height}})`;
}}

document.getElementById('prev').addEventListener('click', () => updateSlice(parseInt(zslider.value, 10) - 1));
document.getElementById('next').addEventListener('click', () => updateSlice(parseInt(zslider.value, 10) + 1));
zslider.addEventListener('input', () => updateSlice(parseInt(zslider.value, 10)));
showAnn.addEventListener('change', () => updateSlice(parseInt(zslider.value, 10)));

updateSlice(0);
</script>
</body>
</html>
"""


def hex_to_rgba_factor(hex_color: str):
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        return [1.0, 0.42, 0.42, 1.0]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b, 1.0]


def build_glb_from_meshes(meshes: list[dict]) -> bytes:
    if not meshes:
        raise ValueError("没有可写入 GLB 的网格")

    bin_blob = bytearray()
    buffer_views = []
    accessors = []
    gltf_meshes = []
    nodes = []
    materials = []

    def pad4(b: bytearray):
        while len(b) % 4 != 0:
            b.append(0)

    for mesh_idx, mesh in enumerate(meshes):
        positions = np.asarray(mesh["positions"], dtype=np.float32)
        normals = np.asarray(mesh["normals"], dtype=np.float32)
        indices = np.asarray(mesh["indices"], dtype=np.uint32).reshape(-1)

        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()

        pos_offset = len(bin_blob)
        pos_bytes = positions.tobytes()
        bin_blob.extend(pos_bytes)
        pad4(bin_blob)

        nor_offset = len(bin_blob)
        nor_bytes = normals.tobytes()
        bin_blob.extend(nor_bytes)
        pad4(bin_blob)

        idx_offset = len(bin_blob)
        idx_bytes = indices.tobytes()
        bin_blob.extend(idx_bytes)
        pad4(bin_blob)

        pos_bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(pos_bytes),
                "target": 34962,
            }
        )
        nor_bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": nor_offset,
                "byteLength": len(nor_bytes),
                "target": 34962,
            }
        )
        idx_bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": len(idx_bytes),
                "target": 34963,
            }
        )

        pos_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": pos_bv_idx,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            }
        )
        nor_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": nor_bv_idx,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            }
        )
        idx_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": idx_bv_idx,
                "componentType": 5125,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            }
        )

        material_idx = len(materials)
        materials.append(
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": hex_to_rgba_factor(mesh.get("color", "#ff6b6b")),
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.8,
                },
                "doubleSided": True,
            }
        )

        gltf_meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc_idx, "NORMAL": nor_acc_idx},
                        "indices": idx_acc_idx,
                        "material": material_idx,
                        "mode": 4,
                    }
                ]
            }
        )
        nodes.append({"mesh": mesh_idx})

    gltf = {
        "asset": {"version": "2.0", "generator": "all_in_one_npz_glb_tool.py"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4 != 0:
        json_chunk += b" "

    while len(bin_blob) % 4 != 0:
        bin_blob.append(0)

    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_blob)

    output = bytearray()
    output.extend(struct.pack("<4sII", b"glTF", 2, total_length))
    output.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    output.extend(json_chunk)
    output.extend(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
    output.extend(bin_blob)
    return bytes(output)


def make_embedded_glb_viewer_html(glb_bytes: bytes, title: str = "Embedded GLB Viewer") -> str:
    b64 = base64.b64encode(glb_bytes).decode("ascii")
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{safe_title}</title>
  <style>
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:#101216; color:#e6e6e6; }}
    #top {{ padding:10px 12px; border-bottom:1px solid #2a2e36; }}
    #view {{ width:100vw; height:calc(100vh - 44px); }}
    .hint {{ color:#b6beca; font-size:13px; }}
  </style>
</head>
<body>
  <div id=\"top\">{safe_title} · <span class=\"hint\">左键旋转，滚轮缩放，右键平移</span></div>
  <div id=\"view\"></div>

<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js\"></script>
<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>
<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js\"></script>
<script>
(function() {{
  const container = document.getElementById('view');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111318);

  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 100000);
  camera.position.set(0, -200, 120);
  camera.up.set(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({{ antialias:true }});
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dl = new THREE.DirectionalLight(0xffffff, 0.8);
  dl.position.set(1, 1, 1);
  scene.add(dl);

  const loader = new THREE.GLTFLoader();
  const b64 = "{b64}";
  const binary = Uint8Array.from(atob(b64), c => c.charCodeAt(0));

  loader.parse(binary.buffer, '', function(gltf) {{
    scene.add(gltf.scene);

    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    gltf.scene.position.sub(center);

    const radius = Math.max(size.x, size.y, size.z) || 1;
    camera.position.set(0, -radius * 2.2, radius * 1.3);
    controls.target.set(0, 0, 0);
    controls.update();
  }}, function(err) {{
    alert('GLB 解析失败: ' + (err && err.message ? err.message : err));
  }});

  window.addEventListener('resize', function() {{
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }});

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();
}})();
</script>
</body>
</html>
"""


def load_npz_volume(npz_files: Sequence[Path], ann_threshold: float):
    raws = []
    anns_bool = []
    anns_raw = []
    shape = None

    for file_path in npz_files:
        with np.load(file_path, allow_pickle=True) as npz_obj:
            raw, ann = detect_arrays(npz_obj)

        if raw is None:
            raise ValueError(f"无法在文件中识别原图数组: {file_path}")

        raw2d = reduce_to_2d(np.asarray(raw))
        if ann is None:
            ann2d_raw = np.zeros_like(raw2d, dtype=np.float32)
            ann2d_bool = np.zeros_like(raw2d, dtype=bool)
        else:
            ann2d = reduce_to_2d(np.asarray(ann))
            ann2d_raw = ann2d.astype(np.float32)
            if ann2d.dtype == bool:
                ann2d_bool = ann2d.astype(bool)
            else:
                ann2d_bool = (ann2d_raw > ann_threshold)

        if shape is None:
            shape = raw2d.shape
        elif raw2d.shape != shape:
            raise ValueError(
                f"切片大小不一致: {file_path} 的 shape={raw2d.shape}, 期望={shape}"
            )

        raws.append(raw2d.astype(np.float32))
        anns_bool.append(ann2d_bool)
        anns_raw.append(ann2d_raw)

    if not raws:
        raise ValueError("没有可用的 NPZ 切片")

    vol_raw = np.stack(raws, axis=0)
    vol_ann_bool = np.stack(anns_bool, axis=0).astype(np.uint8)
    vol_ann_raw = np.stack(anns_raw, axis=0).astype(np.float32)
    return vol_raw, vol_ann_bool, vol_ann_raw


def extract_meshes_from_ann(vol_ann_raw: np.ndarray) -> list[dict]:
    if marching_cubes is None:
        raise RuntimeError("缺少 scikit-image：无法进行 marching_cubes")

    meshes = []

    def one_mask_to_mesh(mask: np.ndarray, color: str):
        if not mask.any():
            return
        verts, faces, normals, _ = marching_cubes(mask.astype(np.uint8), level=0.5)
        verts_xyz = verts[:, [2, 1, 0]].astype(np.float32)
        normals_xyz = normals[:, [2, 1, 0]].astype(np.float32)
        meshes.append(
            {
                "positions": verts_xyz,
                "normals": normals_xyz,
                "indices": faces.astype(np.uint32),
                "color": color,
            }
        )

    yellow_mask = vol_ann_raw > 1.0
    red_mask = (vol_ann_raw > 0.0) & (vol_ann_raw <= 1.0)

    one_mask_to_mesh(yellow_mask, "#ffd400")
    one_mask_to_mesh(red_mask, "#ff3b3b")

    return meshes


def convert_npz_set(npz_files: Sequence[Path], settings: Mode1Settings) -> list[Path]:
    npz_files = sorted([p.resolve() for p in npz_files], key=lambda x: natural_sort_key(str(x)))
    vol_raw, _, vol_ann_raw = load_npz_volume(npz_files, settings.ann_threshold)

    vol_raw_u8 = normalize_u8(vol_raw)
    z_count, h, w = vol_raw_u8.shape

    outputs: list[Path] = []
    stem = npz_files[0].parent.name + "_" + npz_files[0].stem + f"_to_{npz_files[-1].stem}"
    stem = re.sub(r"[^0-9a-zA-Z_\-]+", "_", stem)

    if settings.export_2d:
        raw_b64 = [to_png_base64(vol_raw_u8[z]) for z in range(z_count)]
        overlay_b64 = [overlay_base64(vol_raw_u8[z], vol_ann_raw[z]) for z in range(z_count)]
        payload = {
            "raw_images": raw_b64,
            "overlay_images": overlay_b64,
            "z_count": int(z_count),
            "width": int(w),
            "height": int(h),
        }
        out_2d_dir = ensure_dir(settings.out_2d)
        out_2d = out_2d_dir / f"{stem}_2d.html"
        out_2d.write_text(make_2d_html(payload), encoding="utf-8")
        outputs.append(out_2d)

    need_mesh = settings.export_glb or settings.export_3d
    glb_bytes = None

    if need_mesh:
        if np.count_nonzero(vol_ann_raw > 0) == 0:
            print("提示：标注体积全零，无法导出 GLB / 3D HTML。")
        else:
            meshes = extract_meshes_from_ann(vol_ann_raw)
            if not meshes:
                print("提示：未提取到有效网格，跳过 GLB / 3D HTML。")
            else:
                glb_bytes = build_glb_from_meshes(meshes)

    if settings.export_glb and glb_bytes is not None:
        out_glb_dir = ensure_dir(settings.out_glb)
        out_glb = out_glb_dir / f"{stem}.glb"
        out_glb.write_bytes(glb_bytes)
        outputs.append(out_glb)

    if settings.export_3d and glb_bytes is not None:
        out_3d_dir = ensure_dir(settings.out_3d)
        out_3d = out_3d_dir / f"{stem}_3d.html"
        out_3d.write_text(make_embedded_glb_viewer_html(glb_bytes, title=f"{stem} 3D Viewer"), encoding="utf-8")
        outputs.append(out_3d)

    return outputs


def convert_glb_files_to_html(glb_files: Sequence[Path], out_dir: str | Path) -> list[Path]:
    out_path = ensure_dir(out_dir)
    outputs: list[Path] = []

    for glb in sorted(glb_files, key=lambda x: natural_sort_key(str(x))):
        glb_bytes = glb.read_bytes()
        html = make_embedded_glb_viewer_html(glb_bytes, title=f"{glb.name} Viewer")
        out_file = out_path / f"{glb.stem}.html"
        out_file.write_text(html, encoding="utf-8")
        outputs.append(out_file)

    return outputs


def ask_yes_no(question: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        user_input = input(f"{question} [{default_text}]: ").strip().lower()
        if user_input == "":
            return default
        if user_input in {"y", "yes", "是", "1", "true"}:
            return True
        if user_input in {"n", "no", "否", "0", "false"}:
            return False
        print("输入无效，请输入 y/n（或直接回车使用默认值）。")


def ask_path(question: str, default_path: str) -> str:
    user_input = input(f"{question}（默认: {default_path}）: ").strip()
    return user_input if user_input else default_path


def choose_mode_interactive(default_mode: int = 1) -> int:
    while True:
        text = input("【0. 选择模式】1. npz转glb/html/带有glb的html，2. 多个glb转html（默认1）: ").strip()
        if text == "":
            return default_mode
        if text in {"1", "2"}:
            return int(text)
        print("请输入 1 或 2。")


def show_mode1_settings(settings: Mode1Settings):
    print("\n当前设置：")
    print(f"- 【1】导出2D HTML：{'是' if settings.export_2d else '否'}")
    if settings.export_2d:
        print(f"  【1.1】2D HTML 目录：{settings.out_2d}")
    print(f"- 【2】导出GLB：{'是' if settings.export_glb else '否'}")
    if settings.export_glb:
        print(f"  【2.1】GLB 目录：{settings.out_glb}")
    print(f"- 【3】导出3D HTML：{'是' if settings.export_3d else '否'}")
    if settings.export_3d:
        print(f"  【3.1】3D HTML 目录：{settings.out_3d}")
    print(f"- 标注阈值 ann_threshold：{settings.ann_threshold}")


def setup_mode1_interactive(settings: Mode1Settings):
    print("\n请进行模式1设置（可直接回车采用默认值）：")
    settings.export_2d = ask_yes_no("【1. 是否导出2d html文件】", settings.export_2d)
    if settings.export_2d:
        settings.out_2d = ask_path("【1.1 导出2d html文件的位置】", settings.out_2d)

    settings.export_glb = ask_yes_no("【2. 是否导出glb文件】", settings.export_glb)
    if settings.export_glb:
        settings.out_glb = ask_path("【2.1 导出glb文件的位置】", settings.out_glb)

    settings.export_3d = ask_yes_no("【3. 同时导出3d html文件】", settings.export_3d)
    if settings.export_3d:
        settings.out_3d = ask_path("【3.1 导出3d html文件的位置】", settings.out_3d)

    ann_text = input(f"【4. 标注阈值 ann_threshold】（默认 {settings.ann_threshold}）: ").strip()
    if ann_text:
        try:
            settings.ann_threshold = float(ann_text)
        except ValueError:
            print("阈值输入无效，保留原值。")


def setup_mode2_interactive(settings: Mode2Settings):
    print("\n模式2设置：")
    settings.out_html = ask_path("【A. 导出html文件的位置】", settings.out_html)


def print_outputs(outputs: Sequence[Path]):
    if not outputs:
        print("本次没有输出文件。")
        return
    print("转换完成，输出文件：")
    for out in outputs:
        print(f"- {out}")


def mode1_loop(settings: Mode1Settings):
    show_mode1_settings(settings)
    print("\n请输入：")
    print("- 多个npz文件路径（空格分割），或")
    print("- 一个包含npz集合的文件夹路径，或")
    print("- s（重新设置），q（退出）")

    while True:
        text = input("\n模式1 输入> ").strip()
        if text.lower() == "q":
            print("已退出。")
            return
        if text.lower() == "s":
            setup_mode1_interactive(settings)
            show_mode1_settings(settings)
            continue

        npz_files = parse_input_to_files(text, ".npz")
        if not npz_files:
            print("未找到有效 npz 文件，请检查输入路径是否存在且包含 .npz。")
            continue

        try:
            outputs = convert_npz_set(npz_files, settings)
            print_outputs(outputs)
        except Exception as exc:
            print(f"转换失败：{exc}")


def mode2_loop(settings: Mode2Settings):
    print(f"\n模式2 当前输出目录：{Path(settings.out_html).expanduser().resolve()}")
    print("请输入：")
    print("- 多个glb文件路径（空格分割），或")
    print("- 一个包含glb集合的文件夹路径，或")
    print("- s（设置），q（退出）")

    while True:
        text = input("\n模式2 输入> ").strip()
        if text.lower() == "q":
            print("已退出。")
            return
        if text.lower() == "s":
            setup_mode2_interactive(settings)
            print(f"已更新输出目录：{Path(settings.out_html).expanduser().resolve()}")
            continue

        glb_files = parse_input_to_files(text, ".glb")
        if not glb_files:
            print("未找到有效 glb 文件，请检查输入路径是否存在且包含 .glb。")
            continue

        try:
            outputs = convert_glb_files_to_html(glb_files, settings.out_html)
            print_outputs(outputs)
        except Exception as exc:
            print(f"转换失败：{exc}")


def apply_args_to_mode1_settings(args, settings: Mode1Settings):
    if args.export_2d is not None:
        settings.export_2d = args.export_2d
    if args.out_2d:
        settings.out_2d = args.out_2d

    if args.export_glb is not None:
        settings.export_glb = args.export_glb
    if args.out_glb:
        settings.out_glb = args.out_glb

    if args.export_3d is not None:
        settings.export_3d = args.export_3d
    if args.out_3d:
        settings.out_3d = args.out_3d

    if args.ann_threshold is not None:
        settings.ann_threshold = args.ann_threshold


def collect_input_files_from_args(args, ext: str) -> list[Path]:
    files: list[Path] = []

    if args.input_files:
        files.extend(gather_paths_from_tokens(args.input_files, ext))

    if args.input_dir:
        files.extend(gather_paths_from_dir(args.input_dir, ext))

    if args.input_list:
        p = Path(args.input_list).expanduser()
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8", errors="ignore").strip()
            tokens = content.split()
            files.extend(gather_paths_from_tokens(tokens, ext))

    dedup = sorted(set(files), key=lambda x: natural_sort_key(str(x)))
    return dedup


def parse_args():
    parser = argparse.ArgumentParser(description="单文件 NPZ/GLB 转换工具")
    parser.add_argument("--mode", choices=["1", "2"], help="运行模式：1=NPZ转换，2=GLB转HTML")

    parser.add_argument("--input-files", nargs="*", help="输入文件列表（空格分割）")
    parser.add_argument("--input-dir", help="输入文件夹（自动递归查找对应后缀）")
    parser.add_argument("--input-list", help="文本文件，内部为空格/换行分割的文件路径")
    parser.add_argument("--once", action="store_true", help="处理完命令行输入后退出，不进入交互")

    parser.add_argument("--export-2d", dest="export_2d", action="store_true", default=None)
    parser.add_argument("--no-export-2d", dest="export_2d", action="store_false")
    parser.add_argument("--out-2d", help="2D HTML 输出目录")

    parser.add_argument("--export-glb", dest="export_glb", action="store_true", default=None)
    parser.add_argument("--no-export-glb", dest="export_glb", action="store_false")
    parser.add_argument("--out-glb", help="GLB 输出目录")

    parser.add_argument("--export-3d", dest="export_3d", action="store_true", default=None)
    parser.add_argument("--no-export-3d", dest="export_3d", action="store_false")
    parser.add_argument("--out-3d", help="3D HTML 输出目录")

    parser.add_argument("--ann-threshold", type=float, help="标注阈值（默认0.5）")
    parser.add_argument("--out-html", help="模式2输出 HTML 目录")

    return parser.parse_args()


def run_mode1_once_from_args(args, settings: Mode1Settings) -> bool:
    npz_files = collect_input_files_from_args(args, ".npz")
    if not npz_files:
        return False

    print(f"命令行模式1：检测到 {len(npz_files)} 个 npz 文件，开始转换...")
    outputs = convert_npz_set(npz_files, settings)
    print_outputs(outputs)
    return True


def run_mode2_once_from_args(args, settings: Mode2Settings) -> bool:
    glb_files = collect_input_files_from_args(args, ".glb")
    if not glb_files:
        return False

    print(f"命令行模式2：检测到 {len(glb_files)} 个 glb 文件，开始转换...")
    outputs = convert_glb_files_to_html(glb_files, settings.out_html)
    print_outputs(outputs)
    return True


def print_welcome():
    print("\n欢迎使用 All-in-One NPZ/GLB 转换工具！")
    print("给你讲个冷笑话：")
    print(random.choice(COLD_JOKES))


def main():
    args = parse_args()
    print_welcome()

    mode1_settings = Mode1Settings()
    mode2_settings = Mode2Settings()

    apply_args_to_mode1_settings(args, mode1_settings)
    if args.out_html:
        mode2_settings.out_html = args.out_html

    if args.mode == "1":
        processed = run_mode1_once_from_args(args, mode1_settings)
        if args.once:
            if not processed:
                print("未检测到可处理的 npz 输入，已退出。")
            return
        if not processed:
            setup_mode1_interactive(mode1_settings)
        mode1_loop(mode1_settings)
        return

    if args.mode == "2":
        processed = run_mode2_once_from_args(args, mode2_settings)
        if args.once:
            if not processed:
                print("未检测到可处理的 glb 输入，已退出。")
            return
        mode2_loop(mode2_settings)
        return

    selected_mode = choose_mode_interactive(default_mode=1)
    if selected_mode == 1:
        setup_mode1_interactive(mode1_settings)
        mode1_loop(mode1_settings)
    else:
        mode2_loop(mode2_settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，程序结束。")
