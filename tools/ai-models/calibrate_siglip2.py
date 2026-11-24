"""
用真实照片标定导出好的 SigLIP 2 一档：敏感画面、“像不像有字”的阈值，顺便核对文本补齐方向。

为什么要它：这两个阈值是零样本分类的概率门槛，只有拿真照片才知道合适的值；随机小模型上跑通的导出流程说明不了。
敏感打码是隐私功能，门槛偏了要么漏打码、要么满屏打码。

准备一个目录，按类别放照片（jpg/png/webp，每类几十张就够，越像用户的相册越好）：
  sensitive/  应该打码的       safe/    不该打码的（泳装、自拍、人像也放进来）
  text/       有字（截图、票据、文档、招牌）   notext/  没字的（人、风景、宠物、食物）
没有的类别可以不放，对应的那项就不标定。

  python calibrate_siglip2.py --model-dir dist --id siglip2-base --images calib/ [--write]

做的事（预处理、提示语、算法都和 App 一样，提示语直接从 Kotlin 源码里读）：
  1. 用清单里的分词规则编码提示语，算图片向量（int8 ONNX），按 App 的 softmax 算概率
  2. 敏感：在保证 sensitive 召回率 ≥ 95% 的前提下取最高的阈值；有字：在 text 召回率 ≥ 95% 的前提下取最高的阈值
  3. 用左补齐、右补齐各做一遍零样本分类，看哪种更准：和清单里的不一样且明显更差就报警（说明导出时的补齐方向和训练时不一致）
  --write：把标定出的阈值写进 manifest-<id>.json 的 thresholds（App 读它；没写时用代码里的默认值）
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEARCH = os.path.join(ROOT, "android", "core", "search", "src", "main", "kotlin", "app", "zyll", "search")
TARGET_RECALL = 0.95


def kotlin_list(path, name):
    """从 Kotlin 源码里读 `val name = listOf("...", ...)`。"""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(rf"val {name} = listOf\((.*?)\n\s*\)", src, re.S)
    if not m:
        raise SystemExit(f"{path} 里找不到 {name}")
    return re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))


def encode(sp, spec, text, pad_left):
    pieces = sp.encode(text.lower() if spec["lowercase"] else text)
    bos, eos = spec["bosId"], spec["eosId"]
    room = spec["length"] - (bos >= 0) - (eos >= 0)
    ids = ([bos] if bos >= 0 else []) + pieces[:room] + ([eos] if eos >= 0 else [])
    pad = [spec["padId"]] * (spec["length"] - len(ids))
    return (pad + ids) if pad_left else (ids + pad)


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def prob(img, pos, neg, scale):
    """和 Kotlin 的 SensitiveContent.probability 一样：两组提示语一起做 softmax，取第一组的概率和。"""
    logits = np.concatenate([pos, neg]) @ img * scale
    e = np.exp(logits - logits.max())
    return float(e[: len(pos)].sum() / e.sum())


def pick(pos_scores, neg_scores):
    """召回率不低于 TARGET_RECALL 时最高的阈值，以及这个阈值下的召回率和误报率。"""
    s = np.sort(np.asarray(pos_scores))
    k = int(np.floor((1 - TARGET_RECALL) * len(s)))
    t = float(s[k]) if len(s) else 0.5
    recall = float(np.mean(np.asarray(pos_scores) >= t))
    fpr = float(np.mean(np.asarray(neg_scores) >= t)) if len(neg_scores) else float("nan")
    return round(t, 4), recall, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    mpath = os.path.join(args.model_dir, f"manifest-{args.id}.json")
    with open(mpath, encoding="utf-8") as f:
        m = json.load(f)
    spec = m["tokenizer"]
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.model_dir, spec["file"]))
    opts = ort.SessionOptions()
    img_s = ort.InferenceSession(os.path.join(args.model_dir, m["image"]), opts, providers=["CPUExecutionProvider"])
    txt_s = ort.InferenceSession(os.path.join(args.model_dir, m["text"]), opts, providers=["CPUExecutionProvider"])
    scale = m["logitScale"]
    size = m["imageSize"]
    mean = np.array(m["mean"], np.float32)[:, None, None]
    std = np.array(m["std"], np.float32)[:, None, None]

    def embed_texts(texts, pad_left):
        ids = np.array([encode(sp, spec, t, pad_left) for t in texts], np.int64)
        return norm(np.concatenate([txt_s.run(None, {"input_ids": ids[i:i + 1]})[0] for i in range(len(ids))]))

    def embed_image(path):
        # App：拉伸成正方形（Bitmap.createScaledBitmap，双线性），(x/255 - mean)/std
        im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        x = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        return norm(img_s.run(None, {"pixel_values": x[None].astype(np.float32)})[0][0])

    groups = {}
    for g in ("sensitive", "safe", "text", "notext"):
        files = sorted(p for ext in ("jpg", "jpeg", "png", "webp") for p in glob.glob(os.path.join(args.images, g, f"*.{ext}")))
        if files:
            groups[g] = np.stack([embed_image(p) for p in files])
            print(f"{g}: {len(files)} 张")

    sc = os.path.join(SEARCH, "SensitiveContent.kt")
    tp = os.path.join(SEARCH, "SmartCategories.kt")
    prompts = {
        "sensitive": (kotlin_list(sc, "sensitivePrompts"), kotlin_list(sc, "safePrompts"), "sensitive", "safe"),
        "text": (kotlin_list(tp, "withText"), kotlin_list(tp, "withoutText"), "text", "notext"),
    }
    pad_left = bool(spec.get("padLeft", False))
    thresholds = dict(m.get("thresholds", {}))

    for key, (pp, nn, pos_g, neg_g) in prompts.items():
        if pos_g not in groups:
            print(f"[{key}] 没有 {pos_g}/ 的照片，跳过")
            continue
        acc = {}
        for side in (pad_left, not pad_left):
            pos_v, neg_v = embed_texts(pp, side), embed_texts(nn, side)
            ps = [prob(v, pos_v, neg_v, scale) for v in groups[pos_g]]
            ns = [prob(v, pos_v, neg_v, scale) for v in groups.get(neg_g, [])]
            # 零样本分类准确率（按 0.5 分）：只用来比较两种补齐方向
            acc[side] = (np.mean(np.array(ps) >= 0.5) * len(ps) + np.mean(np.array(ns) < 0.5) * len(ns) if ns else
                         np.mean(np.array(ps) >= 0.5) * len(ps)) / (len(ps) + len(ns))
            if side == pad_left:
                t, recall, fpr = pick(ps, ns)
                print(f"[{key}] 阈值 {t}：{pos_g} 召回 {recall:.0%}，{neg_g} 误报 {fpr:.0%}"
                      f"（现在用的 {thresholds.get(key if key == 'sensitive' else 'text', '代码默认值')}）")
                thresholds["sensitive" if key == "sensitive" else "text"] = t
        side_name = {True: "左补齐", False: "右补齐"}
        print(f"[{key}] 零样本准确率：{side_name[pad_left]}（清单里的）{acc[pad_left]:.0%}，{side_name[not pad_left]} {acc[not pad_left]:.0%}")
        if acc[not pad_left] > acc[pad_left] + 0.1:
            print(f"  警告：换成{side_name[not pad_left]}明显更准，导出时的补齐方向可能和训练时不一致，先查清楚再发布")

    if args.write:
        m["thresholds"] = thresholds
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print("已写入", mpath, thresholds)


if __name__ == "__main__":
    main()
