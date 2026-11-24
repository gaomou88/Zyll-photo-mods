"""
准备 PP-OCRv5 移动端（Apache 2.0，https://github.com/PaddlePaddle/PaddleOCR）的文字检测、识别模型，
和搜索模型放在同一档里，App 用它识别照片里的文字。

从 Hugging Face 上 PaddlePaddle 官方的 ONNX 版本下载：
  PaddlePaddle/PP-OCRv5_mobile_det_onnx   文字检测（DB）
  PaddlePaddle/PP-OCRv5_mobile_rec_onnx   文字识别（CTC），字表在 inference.yml 里

输出到 --out：
  ppocrv5-det.onnx  ppocrv5-rec.onnx  ppocrv5-dict.txt
  ocr.json  前后处理参数（从官方 inference.yml 读出来）+ 三个文件的大小和 sha256，export_siglip2.py 会把它并进清单

最后用和 App（core/ai 的 OcrEngine）一样的前后处理识别一张合成的图，认不出来就失败。

用法：python prepare_ppocr.py --out dist/ocr
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil

import numpy as np
import onnxruntime as ort
import yaml
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

DET_REPO = "PaddlePaddle/PP-OCRv5_mobile_det_onnx"
REC_REPO = "PaddlePaddle/PP-OCRv5_mobile_rec_onnx"
# 识别结果按这个 key 存；换了模型或参数就改它，App 会重新识别
OCR_KEY = "ppocrv5-mobile@1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_key(node, key):
    """在 yml 里递归找第一个叫 key 的值。"""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def onnx_in(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*.onnx"), recursive=True))
    if not files:
        raise SystemExit(f"{d} 里没有 .onnx")
    return files[0]


def yml_in(d):
    files = sorted(glob.glob(os.path.join(d, "**", "inference.yml"), recursive=True))
    if not files:
        return {}
    with open(files[0], encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Ocr:
    """和 App 里 OcrEngine 一样的前后处理（外接矩形框、BGR、CTC 贪心解码）。"""

    def __init__(self, det, rec, chars, p):
        self.det = ort.InferenceSession(det, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(rec, providers=["CPUExecutionProvider"])
        self.chars = chars + [" "]
        self.p = p

    def __call__(self, img):
        p = self.p
        rgb = img.convert("RGB")
        W, H = rgb.size
        k = min(1.0, p["detMaxSide"] / max(W, H))
        w = max(32, round(W * k / 32) * 32)
        h = max(32, round(H * k / 32) * 32)
        small = np.asarray(rgb.resize((w, h), Image.BILINEAR)).astype(np.float32)[:, :, ::-1] / 255.0
        x = ((small - np.array(p["detMean"])) / np.array(p["detStd"])).transpose(2, 0, 1)[None].astype(np.float32)
        prob = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        lab, _ = ndimage.label(prob > p["detThresh"], structure=np.ones((3, 3)))
        out = []
        for sl in ndimage.find_objects(lab):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            bw, bh = x1 - x0, y1 - y0
            if min(bw, bh) < 3 or prob[y0:y1, x0:x1].mean() < p["boxThresh"]:
                continue
            d = bw * bh * p["unclipRatio"] / (2 * (bw + bh))
            fx, fy = W / w, H / h
            box = (max(0, int((x0 - d) * fx)), max(0, int((y0 - d) * fy)),
                   min(W, math.ceil((x1 + d) * fx)), min(H, math.ceil((y1 + d) * fy)))
            crop = rgb.crop(box)
            if crop.height >= crop.width * 1.5:
                crop = crop.rotate(90, expand=True)
            rw = max(1, math.ceil(p["recHeight"] * crop.width / crop.height))
            tw = max(p["recMinWidth"], rw)
            c = np.asarray(crop.resize((rw, p["recHeight"]), Image.BILINEAR)).astype(np.float32)[:, :, ::-1] / 127.5 - 1
            buf = np.zeros((3, p["recHeight"], tw), np.float32)
            buf[:, :, :rw] = c.transpose(2, 0, 1)
            probs = self.rec.run(None, {self.rec.get_inputs()[0].name: buf[None]})[0][0]
            idx, mx = probs.argmax(1), probs.max(1)
            text, scores, prev = [], [], 0
            for i, m in zip(idx, mx):
                if i != 0 and i != prev and i - 1 < len(self.chars):
                    text.append(self.chars[i - 1])
                    scores.append(m)
                prev = i
            if scores and np.mean(scores) >= p["minScore"]:
                out.append((box[1], box[0], "".join(text).strip()))
        return [t for _, _, t in sorted(out)]


def font(size, cjk):
    names = (["NotoSansCJK-Regular.ttc", "wqy-zenhei.ttc", "NotoSansSC-Regular.otf"] if cjk else []) + ["DejaVuSans.ttf"]
    for root in ["/usr/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]:
        for n in names:
            hits = glob.glob(os.path.join(root, "**", n), recursive=True)
            if hits:
                return ImageFont.truetype(hits[0], size)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det_dir = snapshot_download(DET_REPO, cache_dir=args.cache)
    rec_dir = snapshot_download(REC_REPO, cache_dir=args.cache)
    det_yml, rec_yml = yml_in(det_dir), yml_in(rec_dir)

    chars = find_key(rec_yml, "character_dict")
    if not chars:
        raise SystemExit("识别模型的 inference.yml 里没有 character_dict")
    chars = [str(c) for c in chars]

    norm = find_key(det_yml, "NormalizeImage") or {}
    post = find_key(det_yml, "PostProcess") or {}
    rec_shape = find_key(rec_yml, "image_shape") or [3, 48, 320]
    params = {
        # 手机上按长边 960 检测：比官方默认省得多，照片里的字一般够用
        "detMaxSide": 960,
        "detMean": [float(v) for v in norm.get("mean", [0.485, 0.456, 0.406])],
        "detStd": [float(v) for v in norm.get("std", [0.229, 0.224, 0.225])],
        "detThresh": float(post.get("thresh", 0.3)),
        "boxThresh": float(post.get("box_thresh", 0.6)),
        "unclipRatio": float(post.get("unclip_ratio", 1.5)),
        "recHeight": int(rec_shape[1]),
        "recMinWidth": int(rec_shape[2]),
        "minScore": 0.5,
    }

    det_out = os.path.join(args.out, "ppocrv5-det.onnx")
    rec_out = os.path.join(args.out, "ppocrv5-rec.onnx")
    dict_out = os.path.join(args.out, "ppocrv5-dict.txt")
    shutil.copyfile(onnx_in(det_dir), det_out)
    shutil.copyfile(onnx_in(rec_dir), rec_out)
    with open(dict_out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chars) + "\n")

    # 识别模型的类别数必须是 空白 + 字表 + 空格
    classes = ort.InferenceSession(rec_out, providers=["CPUExecutionProvider"]).get_outputs()[0].shape[-1]
    if isinstance(classes, int) and classes != len(chars) + 2:
        raise SystemExit(f"字表 {len(chars)} 个字，模型输出 {classes} 类，对不上")

    # ---- 自检：合成一张图，用和 App 一样的流程识别 ----
    ocr = Ocr(det_out, rec_out, chars, params)
    img = Image.new("RGB", (1280, 800), (232, 224, 205))
    d = ImageDraw.Draw(img)
    latin = font(56, cjk=False)
    if latin is None:
        raise SystemExit("找不到字体，没法做自检")
    d.text((80, 120), "ZYLL PHOTO 2026", font=latin, fill=(20, 20, 20))
    cjk = font(56, cjk=True)
    if cjk is not None:
        d.text((80, 420), "发票合计 128.50", font=cjk, fill=(180, 30, 30))
    lines = ocr(img)
    print("self-check:", lines)
    joined = "".join(lines).replace(" ", "").upper()
    if "ZYLL" not in joined or "2026" not in joined:
        raise SystemExit("自检失败：没认出英文那一行")
    if cjk is not None and "发票" not in joined:
        raise SystemExit("自检失败：没认出中文那一行")

    files = []
    for p in (det_out, rec_out, dict_out):
        files.append({"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)})
        print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")
    info = {
        "key": OCR_KEY,
        "det": os.path.basename(det_out),
        "rec": os.path.basename(rec_out),
        "dict": os.path.basename(dict_out),
        **params,
        "license": "PP-OCRv5 (Apache-2.0) https://github.com/PaddlePaddle/PaddleOCR",
        "files": files,
    }
    with open(os.path.join(args.out, "ocr.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
