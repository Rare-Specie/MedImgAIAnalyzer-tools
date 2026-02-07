#!/usr/bin/env python3
"""npz_to_web.py

说明：此脚本将把按切片的 `.npz` 文件（每个包含原始图像和可选标注）合并为一个可在浏览器中打开的 `viewer.html`，支持切片选择、标注开关与 3D 重建查看（若安装 scikit-image）。

将按顺序的 2D .npz 切片（每个包含原始图像和标注）转换为单个可在浏览器中打开的 `viewer.html`。
功能：
 - 自动从 .npz 中检测原始图像与标注（支持常见键名：image, img, raw, data / label, mask, seg）
 - 将切片按文件名排序并堆叠为 3D 体积
 - 生成每层的 base64 PNG（原始 / 带标注 overlay）并把数据嵌入到单个 HTML（或保存为文件并引用）
 - 通过 marching_cubes 从标注体积提取网格，并把顶点与面片嵌入 HTML，以便 Three.js 显示 3D 重建

用法:
    python npz_to_web.py --files files.txt --out viewer.html --embed

依赖:
    pip install numpy pillow scikit-image

快速开始
1. 安装依赖
pip install numpy pillow scikit-image
2. 准备 `files.txt`，内容为每个 `.npz` 文件的绝对或相对路径（空格或换行分隔）。
3. 运行脚本（默认将生成 `viewer.html` 并把图像嵌入其中）

选项
--no-embed：不把图片 base64 嵌入 HTML（未来可扩展为引用外部切片文件）
--ann-threshold：当标注是概率时使用的阈值（默认 0.5）
--skip-marching：跳过 3D 网格生成

注意
 - 脚本通过 heuristics 自动检测 `.npz` 中的原始数组与标注数组（常用键名：`image/img/raw` 与 `label/mask/seg`）。如果自动检测失败，请看脚本并在需要时修改。 
 - 生成的 `viewer.html` 可直接通过浏览器打开（file://），无需服务器。若网格未生成，请确认安装了 `scikit-image`。

"""

import argparse
import json
import os
import sys
import base64
from io import BytesIO

import numpy as np
from PIL import Image

try:
    from skimage.measure import marching_cubes
except Exception:
    marching_cubes = None

# 常见 key 优先级
RAW_KEYS = ["image", "img", "raw", "ct", "data", "slice", "input"]
ANN_KEYS = ["label", "mask", "seg", "annotation", "gt"]


def natural_sort_key(s):
    # simple natural sort key
    import re
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def detect_arrays(npz):
    """从 npz 文件中找出 raw 和 ann 数组"""
    files = list(npz.files)
    # try keys first
    raw = ann = None
    for k in RAW_KEYS:
        if k in files:
            raw = npz[k]
            break
    for k in ANN_KEYS:
        if k in files:
            ann = npz[k]
            break

    # fallback heuristics
    arrays = [(k, npz[k]) for k in files]
    if raw is None:
        # choose first numeric 2D/2D-like array (not boolean)
        for k, v in arrays:
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.dtype != bool:
                raw = v
                break
    if ann is None:
        # choose first boolean or small-int array (0/1) not equal to raw
        for k, v in arrays:
            if v is raw:
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 2 and (v.dtype == bool or np.issubdtype(v.dtype, np.integer)):
                # check unique values
                uv = np.unique(v)
                if uv.size <= 5:
                    ann = v
                    break
    return raw, ann


def to_png_base64(arr):
    """将 numpy 灰度数组（2D）转换为 base64 PNG 字符串（L 模式）"""
    arr_u8 = arr.astype(np.uint8)
    img = Image.fromarray(arr_u8, mode='L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/png;base64,' + encoded


def overlay_base64(raw_u8, ann, alpha=0.4):
    """把标注以半透明颜色叠加到灰度 raw 图上，输出 base64 PNG。"""
    # raw_u8: 2D uint8
    rgb = np.stack([raw_u8] * 3, axis=-1)
    overlay = rgb.copy().astype(np.float32)
    ann = ann.astype(np.float32, copy=False)

    yellow_mask = ann > 1.0
    red_mask = (ann > 0.0) & (ann <= 1.0)

    if yellow_mask.any():
        overlay[yellow_mask, 0] = overlay[yellow_mask, 0] * (1 - alpha) + 255 * alpha
        overlay[yellow_mask, 1] = overlay[yellow_mask, 1] * (1 - alpha) + 255 * alpha
        overlay[yellow_mask, 2] = overlay[yellow_mask, 2] * (1 - alpha) + 0 * alpha

    if red_mask.any():
        overlay[red_mask, 0] = overlay[red_mask, 0] * (1 - alpha) + 255 * alpha
        overlay[red_mask, 1] = overlay[red_mask, 1] * (1 - alpha) + 0 * alpha
        overlay[red_mask, 2] = overlay[red_mask, 2] * (1 - alpha) + 0 * alpha

    im = Image.fromarray(overlay.astype(np.uint8), mode='RGB')
    buf = BytesIO()
    im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', '-f', help='文本文件，列出所有 npz 文件（支持空格或换行分隔）', required=True)
    p.add_argument('--out', '-o', help='输出 HTML 文件路径', default='viewer.html')
    p.add_argument('--embed', action='store_true', help='把图片与网格直接嵌入单个 HTML（默认：嵌入）')
    p.add_argument('--no-embed', dest='embed', action='store_false')
    p.add_argument('--ann-threshold', type=float, default=0.5, help='当标注为概率时的阈值（默认 0.5）')
    p.add_argument('--skip-marching', action='store_true', help='不计算三维重建网格（跳过 marching cubes）')
    p.set_defaults(embed=True)
    args = p.parse_args()

    with open(args.files, 'r') as f:
        txt = f.read().strip()
    if not txt:
        print('files list is empty', file=sys.stderr)
        sys.exit(1)
    paths = [p for p in txt.split() if p.strip()]
    paths = sorted(paths, key=natural_sort_key)
    print(f'Found {len(paths)} files')

    raws = []
    anns = []
    anns_raw = []
    shape = None
    for path in paths:
        if not os.path.exists(path):
            print(f'警告：文件不存在 {path}', file=sys.stderr)
            continue
        npz = np.load(path, allow_pickle=True)
        raw, ann = detect_arrays(npz)
        if raw is None:
            print(f'错误：无法在 {path} 中检测到 raw 数组, keys={npz.files}', file=sys.stderr)
            sys.exit(1)
        # take a 2D slice
        if raw.ndim > 2:
            # if shape[0] is channel, try to squeeze
            # choose the first 2D plane with largest area
            for arr in (raw,):
                # try to reduce to 2D by taking first channel if shape[0] <=4
                if raw.ndim == 3 and raw.shape[0] <= 4:
                    raw2 = raw[0]
                    raw = raw2
                    break
                elif raw.ndim == 3 and raw.shape[-1] <=4:
                    raw = raw[...,0]
                    break
                else:
                    raw = np.squeeze(raw)
                    if raw.ndim == 2:
                        break
        if ann is not None and ann.ndim > 2:
            if ann.ndim == 3 and ann.shape[0] <= 4:
                ann = ann[0]
            elif ann.ndim == 3 and ann.shape[-1] <= 4:
                ann = ann[...,0]
            else:
                ann = np.squeeze(ann)
        # confirm shape
        if raw.ndim != 2:
            print(f'错误：{path} 中 raw 不是 2D 数组 (shape={raw.shape})', file=sys.stderr)
            sys.exit(1)
        if shape is None:
            shape = raw.shape
        else:
            if raw.shape != shape:
                print(f'错误：切片大小不一致 {path} raw.shape={raw.shape} expected={shape}', file=sys.stderr)
                sys.exit(1)
        raws.append(raw.astype(np.float32))
        if ann is None:
            anns.append(np.zeros_like(raw, dtype=bool))
            anns_raw.append(np.zeros_like(raw, dtype=np.float32))
        else:
            if ann.dtype == bool:
                anns.append(ann.astype(bool))
                anns_raw.append(ann.astype(np.float32))
            else:
                anns.append((ann > args.ann_threshold).astype(bool))
                anns_raw.append(ann.astype(np.float32))

    if len(raws) == 0:
        print('没有有效切片', file=sys.stderr)
        sys.exit(1)

    vol_raw = np.stack(raws, axis=0)  # shape (Z, H, W)
    vol_ann = np.stack(anns, axis=0).astype(np.uint8)

    # normalize raw to 0-255
    mmin, mmax = float(np.nanmin(vol_raw)), float(np.nanmax(vol_raw))
    print(f'raw volume range: {mmin} - {mmax}')
    if mmax > mmin:
        vol_raw_u8 = ((vol_raw - mmin) / (mmax - mmin) * 255.0).astype(np.uint8)
    else:
        vol_raw_u8 = np.clip(vol_raw, 0, 255).astype(np.uint8)

    # create per-slice base64 images
    print('生成切片图像（base64）...')
    raw_b64 = [to_png_base64(vol_raw_u8[z]) for z in range(vol_raw_u8.shape[0])]
    vol_ann_raw = np.stack(anns_raw, axis=0).astype(np.float32)
    overlay_b64 = [overlay_base64(vol_raw_u8[z], vol_ann_raw[z]) for z in range(vol_raw_u8.shape[0])]

    mesh = None
    if (not args.skip_marching) and marching_cubes is not None and vol_ann.sum() > 0:
        print('计算 marching cubes 网格（可能需要一些时间）...')
        meshes = []
        yellow_mask = vol_ann_raw > 1.0
        red_mask = (vol_ann_raw > 0.0) & (vol_ann_raw <= 1.0)

        if yellow_mask.any():
            verts, faces, normals, values = marching_cubes(yellow_mask.astype(np.uint8), level=0.5)
            meshes.append({'verts': verts.tolist(), 'faces': faces.tolist(), 'color': '#ffd400'})
            print(f'黄色网格顶点数: {len(verts)}, 面片数: {len(faces)}')

        if red_mask.any():
            verts, faces, normals, values = marching_cubes(red_mask.astype(np.uint8), level=0.5)
            meshes.append({'verts': verts.tolist(), 'faces': faces.tolist(), 'color': '#ff3b3b'})
            print(f'红色网格顶点数: {len(verts)}, 面片数: {len(faces)}')

        if meshes:
            mesh = meshes
    else:
        if marching_cubes is None:
            print('skimage.measure.marching_cubes 未安装，跳过 3D 网格')
        elif vol_ann.sum() == 0:
            print('标注体积全零，跳过 3D 网格')

    # build html
    print('生成 HTML 文件...')
    html_template = HTML_TEMPLATE

    payload = {
        'raw_images': raw_b64,
        'overlay_images': overlay_b64,
        'z_count': vol_raw_u8.shape[0],
        'width': shape[1],
        'height': shape[0],
        'mesh': mesh
    }

    out_html = html_template.replace('/*__PAYLOAD_JSON__*/', json.dumps(payload))

    with open(args.out, 'w') as f:
        f.write(out_html)

    print(f'完成。输出: {args.out} 。用浏览器打开即能查看（无需额外服务器）。')


HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NPZ Volume Viewer</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 16px; }
    #viewer { display:flex; gap:16px; }
    #sliceView { flex:1; }
    #controls { margin-top:8px; }
    img.slice { max-width: 100%; height: auto; image-rendering: pixelated; border:1px solid #ddd; }
    .toolbar { display:flex; gap:8px; align-items:center; }
    #threeContainer { width:600px; height:600px; border:1px solid #ccc; }
  </style>
</head>
<body>
  <h2>NPZ Volume Viewer</h2>
  <div id="viewer">
    <div id="sliceView">
      <div class="toolbar">
        <button id="prev">Prev</button>
        <input id="zslider" type="range" min="0" max="0" value="0" />
        <button id="next">Next</button>
        <label><input id="showAnn" type="checkbox" checked /> Show annotation</label>
      </div>
      <div id="imgWrap" style="margin-top:8px;">
        <img id="sliceImg" class="slice" src="" />
      </div>
      <div id="info" style="margin-top:6px;font-size:90%"></div>
    </div>
    <div>
      <div style="margin-bottom:8px">
        <button id="show3D">Show 3D reconstruction</button>
      </div>
      <div id="threeContainer"></div>
    </div>
  </div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// Payload injected by Python
var PAYLOAD = /*__PAYLOAD_JSON__*/;

var raw_images = PAYLOAD.raw_images;
var overlay_images = PAYLOAD.overlay_images;
var zCount = PAYLOAD.z_count;
var width = PAYLOAD.width;
var height = PAYLOAD.height;
var mesh = PAYLOAD.mesh;

var zslider = document.getElementById('zslider');
var sliceImg = document.getElementById('sliceImg');
var showAnn = document.getElementById('showAnn');
var info = document.getElementById('info');
var prevBtn = document.getElementById('prev');
var nextBtn = document.getElementById('next');

zslider.max = Math.max(0, zCount-1);

function updateSlice(z){
  z = Math.min(Math.max(0, z|0), zCount-1);
  zslider.value = z;
  if (showAnn.checked) sliceImg.src = overlay_images[z];
  else sliceImg.src = raw_images[z];
  info.innerText = `slice ${z+1}/${zCount}  (${width}x${height})`;
}

zslider.addEventListener('input', function(){ updateSlice(parseInt(this.value)); });
prevBtn.addEventListener('click', function(){ updateSlice(parseInt(zslider.value)-1); });
nextBtn.addEventListener('click', function(){ updateSlice(parseInt(zslider.value)+1); });
showAnn.addEventListener('change', function(){ updateSlice(parseInt(zslider.value)); });

updateSlice(0);

// 3D view using Three.js if mesh available
var threeContainer = document.getElementById('threeContainer');
var show3Dbtn = document.getElementById('show3D');
var threeStarted = false;
show3Dbtn.addEventListener('click', function(){
    if (!mesh){ alert('No mesh extracted (no annotation or marching_cubes missing)'); return; }
  if (threeStarted) return;
  threeStarted = true;
  initThree(mesh);
});

function initThree(mesh){
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
  var renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
  threeContainer.appendChild(renderer.domElement);

    var meshes = Array.isArray(mesh) ? mesh : [mesh];
    var meshObjs = [];
    var center = new THREE.Vector3(0, 0, 0);
    var firstCenter = true;

    meshes.forEach(function(m){
        var geometry = new THREE.BufferGeometry();
        var verts = new Float32Array(m.verts.flat());
        var faces = new Uint32Array(m.faces.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geometry.setIndex(new THREE.BufferAttribute(faces, 1));
        geometry.computeVertexNormals();

        geometry.computeBoundingBox();
        if (geometry.boundingBox && firstCenter){
            geometry.boundingBox.getCenter(center);
            firstCenter = false;
        }

        var material = new THREE.MeshStandardMaterial({color:m.color || 0xff6b6b, metalness:0.1, roughness:0.8, side:THREE.DoubleSide});
        var meshObj = new THREE.Mesh(geometry, material);
        meshObjs.push(meshObj);
        scene.add(meshObj);
    });
  var light = new THREE.DirectionalLight(0xffffff,1);
  light.position.set(1,1,1);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0x404040));

  camera.position.set(0, -Math.max(width,height, zCount)*1.5, Math.max(width,height)*1.2);
  camera.up.set(0,0,1);
  camera.lookAt(new THREE.Vector3(0,0,0));

    var rotX = 0;
    var rotY = 0;
    var panX = 0;
    var panY = 0;
    var isDragging = false;
    var lastX = 0;
    var lastY = 0;
    var isPanning = false;

    function applyTransforms(){
        meshObjs.forEach(function(meshObj){
            meshObj.rotation.x = rotX;
            meshObj.rotation.z = rotY;
            meshObj.position.x = -center.x + panX;
            meshObj.position.y = -center.y + panY;
        });
    }

    renderer.domElement.addEventListener('mousedown', function(e){
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        isPanning = e.shiftKey || e.button === 2;
    });
    renderer.domElement.addEventListener('mousemove', function(e){
        if (!isDragging) return;
        var dx = e.clientX - lastX;
        var dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        if (isPanning){
            panX += dx * 0.5;
            panY -= dy * 0.5;
        } else {
            rotY += dx * 0.01;
            rotX += dy * 0.01;
        }
        applyTransforms();
    });
    window.addEventListener('mouseup', function(){ isDragging = false; });
    renderer.domElement.addEventListener('contextmenu', function(e){ e.preventDefault(); });
    renderer.domElement.addEventListener('wheel', function(e){
        e.preventDefault();
        var delta = e.deltaY > 0 ? 1.1 : 0.9;
        camera.position.multiplyScalar(delta);
    }, {passive:false});

    applyTransforms();

    function animate(){ requestAnimationFrame(animate); renderer.render(scene,camera); }
    animate();
}

</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
