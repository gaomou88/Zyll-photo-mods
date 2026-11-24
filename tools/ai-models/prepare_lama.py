"""
准备修图里 AI 橡皮擦的“精细”模型：Big-LaMa（Apache 2.0，https://github.com/advimman/lama），
用 Carve 转好的 ONNX（https://huggingface.co/Carve/LaMa-ONNX，固定 512×512 输入）。
MI-GAN 小而快；LaMa 大一些（约 200 MB），擦大块东西、补背景纹理更自然。

输出到 --out：
  lama-512.onnx
  editor-catalog.json  有 --catalog（已有的修图模型列表）就在它的基础上加上 / 换掉 LaMa 这一项

模型的输出范围（0～1 还是 0～255）用一张合成图实测，写进 outputScale，App 按它换算。
还会检查：擦掉一块后，补上的颜色接近周围的颜色（模型能正常工作）。

用法：python prepare_lama.py --out dist-editor [--catalog 旧的 editor-catalog.json]
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "Carve/LaMa-ONNX"
FILE = "lama_fp32.onnx"
OUT_NAME = "lama-512.onnx"
ENTRY = {
    "id": "big-lama-512",
    "tool": "eraser",
    "title": "精细（Big-LaMa）",
    "summary": "擦大块东西、补背景纹理更自然，速度比快速档慢，建议 6GB 以上内存",
    "license": "Big-LaMa (Apache-2.0) https://github.com/advimman/lama; ONNX: https://huggingface.co/Carve/LaMa-ONNX",
    "minRamMb": 6 * 1024,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = hf_hub_download(REPO, FILE, cache_dir=args.cache)
    dst = os.path.join(args.out, OUT_NAME)
    shutil.copyfile(src, dst)

    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in s.get_inputs()}
    mask_name = next(n for n in inputs if "mask" in n.lower())
    image_name = next(n for n in inputs if n != mask_name)
    shape = inputs[image_name].shape
    size = shape[2] if isinstance(shape[2], int) else 512
    print("inputs:", {n: (i.shape, i.type) for n, i in inputs.items()})
    if inputs[image_name].type != "tensor(float)":
        raise SystemExit("App 按 float 输入喂 LaMa，这个模型的输入类型不对")

    # 合成图：左右两种颜色的平滑渐变，中间挖一块，看补上的颜色对不对（和 App 一样：0～1 的 RGB，遮罩 1 = 要擦）
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([0.2 + 0.6 * xx, 0.5 + 0.2 * yy, 0.7 - 0.4 * xx])[None].astype(np.float32)
    mask = np.zeros((1, 1, size, size), np.float32)
    a, b = size * 3 // 8, size * 5 // 8
    mask[:, :, a:b, a:b] = 1
    out = s.run(None, {image_name: img, mask_name: mask})[0]
    peak = float(out.max())
    scale = 255.0 if peak > 1.5 else 1.0
    filled = out[0, :, a:b, a:b] / scale
    want = img[0, :, a:b, a:b]
    err = float(np.abs(filled - want).mean())
    print(f"output peak {peak:.2f} → outputScale {scale:g}; mean error in the hole {err:.3f}")
    if err > 0.08:
        raise SystemExit("自检失败：补上的颜色和周围差太多")

    entry = dict(ENTRY, file=OUT_NAME, size=os.path.getsize(dst), sha256=sha256(dst), outputScale=scale)
    catalog = {"models": []}
    if args.catalog:
        with open(args.catalog, encoding="utf-8") as f:
            catalog = json.load(f)
    catalog["models"] = [m for m in catalog["models"] if m.get("id") != entry["id"]] + [entry]
    with open(os.path.join(args.out, "editor-catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)
    print(json.dumps(entry, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
