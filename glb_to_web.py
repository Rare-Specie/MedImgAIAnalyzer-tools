#!/usr/bin/env python3
"""glb_to_web.py

将单个 .glb 文件打包成可直接用浏览器打开的 HTML 查看器（基于 Three.js）。

特点：
 - 交互式：若未提供路径，运行后会提示输入文件路径
 - 默认输出：HTML 文件放在源 .glb 同一目录，文件名与源文件同名（后缀 .html）
 - 支持嵌入（base64）或引用外部 .glb
 - 可选择运行后自动在默认浏览器中打开

用法示例：
    python glb_to_web.py                 # 交互式输入
    python glb_to_web.py /path/to/model.glb --embed --open
    python glb_to_web.py model.glb --no-embed --out ./viewer.html

注意：对于非常大的 .glb（示例阈值 20 MB），嵌入会产生很大的 HTML 文件；脚本会警告并要求确认。

参考：基于你的 `npz_to_web.py` 的交互与默认输出位置逻辑。
"""

from __future__ import annotations
import argparse
import base64
import os
import sys
import webbrowser
from pathlib import Path

HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #111; color: #eee }}
    #ui {{ position: absolute; left: 12px; top: 12px; z-index: 10; background: rgba(0,0,0,0.4); padding:8px; border-radius:6px }}
    #canvas{{ width:100vw; height:100vh; display:block }}
    a, button {{ color: #9cf }}
  </style>
</head>
<body>
  <div id="ui">
    <div><strong>{title}</strong></div>
    <div style="margin-top:6px">Source: <code>{src_name}</code></div>
      <div style="margin-top:6px">
        <button id="downloadBtn">Download model</button>
        <button id="chooseBtn" style="display:none">Choose local file</button>
        <label><input id="wire" type="checkbox"/> Wireframe</label>
      </div>
      <input id="fileInput" type="file" accept=".glb,.gltf,model/gltf-binary" style="display:none" />
      <div id="loadHint" style="margin-top:6px;font-size:90%;color:#ccc;display:none;white-space:pre-wrap"></div>
    <div style="margin-top:6px;font-size:90%">Drag to rotate · Scroll to zoom · Shift+drag to pan</div>
  </div>
  <canvas id="canvas"></canvas>

  <!-- Three.js + extras (r128 compatible non-module builds) -->
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>

  <script>
  const EMBED = {embed};
  const MODEL_NAME = {model_name_js};
  const MODEL_B64 = {model_b64_js};
  const MODEL_PATH = {model_path_js};

  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);
  const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 10000);
  camera.position.set(0, 2.5, 5);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0,1,0);
  controls.update();

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5,10,7.5);
  scene.add(dir);

  let modelGroup = new THREE.Group();
  scene.add(modelGroup);

  function resize(){
    const w = innerWidth, h = innerHeight;
    camera.aspect = w/h; camera.updateProjectionMatrix();
    renderer.setSize(w,h);
  }
  window.addEventListener('resize', resize, {passive:true});
  resize();

  function setWireframe(enabled){
    modelGroup.traverse((c)=>{ if(c.isMesh) c.material.wireframe = enabled; });
  }
  document.getElementById('wire').addEventListener('change', (e)=> setWireframe(e.target.checked));

  const loader = new THREE.GLTFLoader();
  function loadArrayBuffer(ab){
    try{
      loader.parse(ab, '', function(gltf){
        modelGroup.add(gltf.scene);
        // center & scale
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = (maxDim > 0) ? (Math.max(size.x, size.y, size.z) ? (Math.max(size.x, size.y, size.z)) : 1) : 1;
        const desired = 2.0;
        gltf.scene.scale.multiplyScalar(desired / Math.max(maxDim, 1e-6));
        gltf.scene.position.sub(center.clone().multiplyScalar(1));
      }, function(err){ alert('模型解析失败：'+err); });
    }catch(err){ alert('加载模型失败：'+err); }
  }

  function b64ToArrayBuffer(b64){
    const bin = atob(b64);
    const len = bin.length;
    const arr = new Uint8Array(len);
    for(let i=0;i<len;i++) arr[i] = bin.charCodeAt(i);
    return arr.buffer;
  }

  function startWithArrayBuffer(ab){
    loadArrayBuffer(ab);
    document.getElementById('downloadBtn').onclick = ()=>{
      const blob = new Blob([ab], {type:'model/gltf-binary'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = MODEL_NAME || 'model.glb'; a.click();
      URL.revokeObjectURL(url);
    };
  }

  function startWithUrl(url){
    // improved error handling: show HTTP status when available and provide actionable hints
    loader.load(url,
      function(gltf){ modelGroup.add(gltf.scene); },
      undefined,
      function(e){
        let msg = '加载失败';
        try{
          if (e && typeof e.status !== 'undefined') msg += ` (status=${e.status})`;
          else if (e && e.target && typeof e.target.status !== 'undefined') msg += ` (status=${e.target.status})`;
          else if (e instanceof ProgressEvent) msg += ' (可能由 file:// 协议或跨域/权限限制引起)';
        }catch(_){ /* ignore */ }

        const hint = document.getElementById('loadHint');
        hint.innerText = msg + '\n\n解决方法（任选其一）：\n 1) 在终端运行 `python3 -m http.server` 然后通过 http://localhost:8000/ 打开该 HTML；\n 2) 在本页面使用 “Choose local file” 按钮直接选择 .glb（不会触发跨域）；\n 3) 使用脚本的 --embed 选项把模型嵌入到 HTML 中。';
        hint.style.display = 'block';
        const chooseBtn = document.getElementById('chooseBtn');
        chooseBtn.style.display = 'inline-block';
        console.error('GLTF load error:', e);
      }
    );

    document.getElementById('downloadBtn').onclick = ()=>{ const a = document.createElement('a'); a.href = url; a.download = MODEL_NAME || 'model.glb'; a.click(); };

    // If page opened via file://, many browsers prevent XHR — proactively show hint and file picker
    try{
      if (location && location.protocol === 'file:'){
        const hint = document.getElementById('loadHint');
        hint.innerText = '注意：通过 file:// 打开时，浏览器可能阻止直接加载本地模型。使用 “Choose local file” 或在模型目录运行本地服务器。';
        hint.style.display = 'block';
        document.getElementById('chooseBtn').style.display = 'inline-block';
      }
    }catch(_){ }
  }

  // File fallback: let user pick a local .glb/.gltf and load via FileReader (works around file:///XHR issues)
  function handleFile(file){
    if (!file) return;
    const hint = document.getElementById('loadHint');
    hint.style.display = 'none';
    const reader = new FileReader();
    reader.onload = function(ev){
      try{ startWithArrayBuffer(ev.target.result); }
      catch(err){ alert('从本地文件加载失败：' + err); }
    };
    reader.onerror = function(){ alert('无法读取所选文件'); };
    reader.readAsArrayBuffer(file);
  }
  document.getElementById('chooseBtn').addEventListener('click', function(){ document.getElementById('fileInput').click(); });
  document.getElementById('fileInput').addEventListener('change', function(ev){ const f = ev.target.files && ev.target.files[0]; if(f) handleFile(f); });

  if (EMBED && MODEL_B64){
    try{
      const ab = b64ToArrayBuffer(MODEL_B64);
      startWithArrayBuffer(ab);
    }catch(e){ alert('嵌入模型解码失败：'+e); }
  } else {
    startWithUrl(MODEL_PATH);
  }

  function animate(){ requestAnimationFrame(animate); renderer.render(scene, camera); }
  animate();
  </script>
</body>
</html>
"""


def make_args():
    p = argparse.ArgumentParser(prog='glb_to_web.py', description='Pack a .glb into a single HTML viewer (interactive by default)')
    p.add_argument('file', nargs='?', help='Path to .glb file. If omitted, you will be prompted.')
    p.add_argument('--out', '-o', help='Output HTML path (default: same dir as source, same basename).')
    p.add_argument('--no-embed', dest='embed', action='store_false', help="Don't embed the .glb in the HTML (copy/reference instead)")
    p.add_argument('--embed', dest='embed', action='store_true', help='Embed the .glb into the HTML as base64 (default)')
    p.add_argument('--open', action='store_true', help='Open the generated HTML in the default browser')
    p.set_defaults(embed=True)
    return p


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt + ' [y/N]: ').strip().lower()
    except EOFError:
        return False
    return ans in ('y', 'yes')


def read_binary(path: Path) -> bytes:
    with path.open('rb') as f:
        return f.read()


def write_out(html: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')


def build_html(model_name: str, src_name: str, embed: bool, model_b64: str | None, model_relpath: str | None) -> str:
    """Safely inject values into the HTML template using explicit replacements

    Avoid using str.format() because the template contains many JavaScript
    object literals with `{}` which collide with Python formatting.
    """
    t = HTML_TEMPLATE
    t = t.replace('{title}', f'GLB Viewer — {model_name}')
    t = t.replace('{src_name}', src_name)
    t = t.replace('{embed}', str(embed).lower())
    t = t.replace('{model_name_js}', repr(model_name))
    t = t.replace('{model_b64_js}', (repr(model_b64) if model_b64 is not None else 'null'))
    t = t.replace('{model_path_js}', (repr(model_relpath) if model_relpath is not None else 'null'))
    return t


def main():
    p = make_args()
    args = p.parse_args()

    glb_path = args.file
    if not glb_path:
        try:
            glb_path = input('Enter path to .glb (or .gltf) file: ').strip() or None
        except EOFError:
            glb_path = None
    if not glb_path:
        print('No input file provided. Exiting.', file=sys.stderr)
        sys.exit(1)

    src = Path(glb_path).expanduser().resolve()
    if not src.exists():
        print(f'File not found: {src}', file=sys.stderr)
        sys.exit(1)

    if src.suffix.lower() not in ('.glb', '.gltf'):
        print('Warning: input does not have .glb/.gltf extension', file=sys.stderr)

    default_out = src.with_suffix('.html')
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out

    # If user chose no-embed but wants output inside source dir, ensure reference is relative
    embed = bool(args.embed)

    size_mb = src.stat().st_size / (1024*1024)
    if embed and size_mb > 20:
        print(f'注意：文件较大 ({size_mb:.1f} MB)。嵌入会生成巨大的 HTML。')
        if not confirm('继续并嵌入吗?'):
            print('中止（请使用 --no-embed 或提供更小的模型）')
            sys.exit(1)

    model_b64 = None
    model_relpath = None
    if embed:
        data = read_binary(src)
        model_b64 = base64.b64encode(data).decode('ascii')
    else:
        # copy or reference: if out_path is in same dir (default) reference by basename; else compute relative path
        if out_path.parent == src.parent:
            model_relpath = src.name
        else:
            try:
                model_relpath = os.path.relpath(src, start=out_path.parent)
            except Exception:
                model_relpath = str(src)

    html = build_html(src.name, str(src), embed, model_b64, model_relpath)
    write_out(html, out_path)
    print(f'✅ 输出: {out_path} (embed={embed})')

    if args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == '__main__':
    main()
