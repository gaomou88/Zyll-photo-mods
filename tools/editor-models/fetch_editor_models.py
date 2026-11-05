"""下载修图用的 AI 模型（魔术橡皮擦），逐个做功能校验，再生成 editor-catalog.json。

每个模型都从原作者 / 公认的转换版下载，先用 ONNX Runtime 真跑一遍：
在灰色背景上放一块白色方块，把方块涂成“要擦掉”，擦完后方块位置应变回接近灰色。
这一步同时确认了输入格式和遮罩的含义（哪个值代表要擦掉），确认不了就不发布。

用法：python fetch_editor_models.py --out dist
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
import onnxruntime as ort

MODELS = [
    {
        "id": "migan-512",
        "tool": "eraser",
        "kind": "migan",  # uint8 图 + uint8 遮罩（0 = 要擦掉），任意尺寸，前后处理都在模型里
        "title": "MI-GAN · 快速",
        "summary": "约 28 MB，几秒出结果，适合去掉路人、杂物这类小东西。",
        "file": "migan_pipeline_v2.onnx",
        "url": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
        "license": "MIT",
        "source": "https://github.com/Picsart-AI-Research/MI-GAN",
        "minRamMb": 0,
    },
    {
        "id": "lama-512",
        "tool": "eraser",
        "kind": "lama",  # float 图 [0,1] + float 遮罩（1 = 要擦掉），固定 512×512
        "title": "LaMa · 精细",
        "summary": "约 200 MB，擦大块区域、带纹理的背景（墙面、草地）更自然，慢一些。",
        "file": "lama_fp32.onnx",
        "url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "license": "Apache-2.0",
        "source": "https://github.com/advimman/lama",
        "minRamMb": 6144,
    },
]

GRAY = 128
WHITE = 255


def download(url, path):
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "zyll-model-fetch"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def names(sess):
    ins = [i.name for i in sess.get_inputs()]
    mask = next((n for n in ins if "mask" in n.lower()), ins[1])
    image = next(n for n in ins if n != mask)
    return image, mask


def scene(size):
    """灰底 + 中间一块白方块；返回 (HWC uint8 图, 方块位置的布尔遮罩)。"""
    img = np.full((size, size, 3), GRAY, np.uint8)
    hole = np.zeros((size, size), bool)
    a, b = size * 3 // 8, size * 5 // 8
    img[a:b, a:b] = WHITE
    hole[a - 4:b + 4, a - 4:b + 4] = True  # 和 App 一样，遮罩比物体略大
    return img, hole


def check_migan(sess):
    image, mask = names(sess)
    for i in sess.get_inputs():
        if i.type != "tensor(uint8)":
            raise SystemExit(f"MI-GAN 输入 {i.name} 应为 uint8，实际 {i.type}")
    size = 384  # 不是 512，确认任意尺寸都能跑
    img, hole = scene(size)
    out = sess.run(None, {
        image: img.transpose(2, 0, 1)[None],
        mask: np.where(hole, 0, 255).astype(np.uint8)[None, None],
    })[0]
    if out.shape != (1, 3, size, size) or out.dtype != np.uint8:
        raise SystemExit(f"MI-GAN 输出形状/类型不对：{out.shape} {out.dtype}")
    return out[0].transpose(1, 2, 0).astype(np.float32), hole, {}


def check_lama(sess):
    image, mask = names(sess)
    for i in sess.get_inputs():
        if i.type != "tensor(float)":
            raise SystemExit(f"LaMa 输入 {i.name} 应为 float，实际 {i.type}")
    shape = sess.get_inputs()[0].shape
    size = shape[2] if isinstance(shape[2], int) else 512
    img, hole = scene(size)
    out = sess.run(None, {
        image: (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
        mask: hole.astype(np.float32)[None, None],
    })[0]
    if out.shape != (1, 3, size, size):
        raise SystemExit(f"LaMa 输出形状不对：{out.shape}")
    scale = 255.0 if float(out.max()) > 1.5 else 1.0  # 有的转换版输出 0~255，有的 0~1
    px = out[0].transpose(1, 2, 0) * (255.0 / scale)
    return px, hole, {"inputSize": int(size), "outputScale": scale}


def verify(m, path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    if len(sess.get_inputs()) != 2:
        raise SystemExit(f"{m['id']}: 应有图和遮罩两个输入")
    px, hole, extra = (check_migan if m["kind"] == "migan" else check_lama)(sess)
    inside = float(np.abs(px[hole] - GRAY).mean())
    outside = float(np.abs(px[~hole] - GRAY).mean())
    print(f"{m['id']}: 擦除区与灰底平均差 {inside:.1f}，区外 {outside:.1f}")
    # 擦掉白方块后应接近灰色（白色和灰差 127）；区外应基本不动
    if inside > 40 or outside > 20:
        raise SystemExit(f"{m['id']}: 功能校验没通过，遮罩含义或输入格式可能不对")
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    entries = []
    for m in MODELS:
        path = os.path.join(args.out, m["file"])
        if not os.path.exists(path):
            print(f"下载 {m['url']}")
            download(m["url"], path)
        extra = verify(m, path)
        entry = {k: m[k] for k in ("id", "tool", "kind", "title", "summary", "file", "license", "source", "minRamMb")}
        entry.update(extra)
        entry["size"] = os.path.getsize(path)
        entry["sha256"] = sha256(path)
        entries.append(entry)
    with open(os.path.join(args.out, "editor-catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "models": entries}, f, ensure_ascii=False, indent=2)
    print(json.dumps(entries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
