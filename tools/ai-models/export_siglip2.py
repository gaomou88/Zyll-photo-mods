"""
把 Google 的 SigLIP 2（Apache 2.0，https://huggingface.co/collections/google/siglip2）导出成手机上用的 ONNX，
连同 PP-OCRv5（prepare_ppocr.py 准备好的）打成 App 里的一档模型。

三档（见 MODELS）：
  base    google/siglip2-base-patch16-224    轻量：4～6GB 内存
  large   google/siglip2-large-patch16-384   主力：8GB
  so400m  google/siglip2-so400m-patch16-384  全能：12GB 以上

每档输出：
  <id>-img.onnx         图像编码器，输入 [1,3,S,S] float32（(x/255-0.5)/0.5，拉伸成正方形），输出 [1,dim]
  <id>-txt.onnx         文本编码器，输入 [1,64] int64，输出 [1,dim]
  <id>-tokenizer.model  Gemma 的 SentencePiece 词表，App 里的 Kotlin 分词器直接读它
  manifest-<id>.json    文件清单、预处理参数、分词规则和几组分词对照（App 加载时自检）、OCR 参数

两个编码器都做 int8 动态量化，再和 PyTorch 原模型逐一对比余弦相似度，偏差过大就失败。
另外用 sentencepiece 逐条对照 Hugging Face 的分词器（App 按 sentencepiece 的规则实现），不一致就失败。
catalog.json（所有档位的列表）由 make_catalog.py 汇总。

用法：python export_siglip2.py --model base --ocr dist/ocr --out dist/
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoProcessor

# 给用户选的几档。minRamMb：建议的手机内存下限（App 用来标“推荐”和提醒）
MODELS = {
    "base": dict(
        repo="google/siglip2-base-patch16-224", id="siglip2-base", version=1, order=0, title="轻量",
        summary="体积最小、速度最快，适合 4～6GB 内存的手机", minRamMb=0,
    ),
    "large": dict(
        repo="google/siglip2-large-patch16-384", id="siglip2-large", version=1, order=1, title="主力",
        summary="识别更准，整理照片稍慢，建议 8GB 以上内存", minRamMb=8 * 1024,
    ),
    "so400m": dict(
        repo="google/siglip2-so400m-patch16-384", id="siglip2-so400m", version=1, order=2, title="全能",
        summary="识别最准，体积大、整理照片较慢，建议 12GB 以上内存的旗舰机", minRamMb=12 * 1024,
    ),
}

# App 加载模型时用来自检分词器的几句话（都是 Python 和 Kotlin 转小写结果一致的字符）
TOKENIZER_TESTS = [
    "a photo of a cat", "Sunset at the BEACH!", "海边的日落", "一只猫在沙发上睡觉", "2024年 生日 蛋糕",
    "東京タワーの夜景", "  two  spaces", "😀 emoji party", "收据 发票 ¥128.50", "한국어 텍스트",
    "a very long sentence " * 12,
]

# 结果截断：相似度换算成 SigLIP 的概率 sigmoid(scale·cos + bias)，低于这个概率的不要
MIN_PROBABILITY = 1e-3
# 离最高分多远就截断，单位是 logit（相当于以前 CLIP 用的 0.06 × 100）
RELATIVE_LOGITS = 6.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cos(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


class ImageEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.get_image_features(pixel_values=pixel_values)


class TextEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m.get_text_features(input_ids=input_ids)


def random_texts(n, seed=1):
    pool = ("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法"
            "照片猫狗海边日落城市夜景花朵汽车蛋糕生日风景山湖雪 abcdefghijklmnopqrstuvwxyz0123456789 .,!?-_/:;()[]<>\t\n"
            "😀🎉ÀéüßΩЖあいうアイウ한글ＡＢＣ１２３")
    rng = random.Random(seed)
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, 60))) for _ in range(n)]


def check_tokenizer(tok, model_file):
    """App 按 sentencepiece 的规则分词：HF 的分词器要和它逐条一致。"""
    sp = spm.SentencePieceProcessor(model_file=model_file)

    def same(t):
        return sp.encode(t.lower()) == tok(t.lower(), add_special_tokens=False)["input_ids"]

    bad_fixed = [t for t in TOKENIZER_TESTS if not same(t)]
    if bad_fixed:
        raise SystemExit(f"tokenizer mismatch: {bad_fixed[0]!r}")
    texts = random_texts(3000)
    bad = [t for t in texts if not same(t)]
    print(f"tokenizer: sentencepiece vs transformers, {len(texts) - len(bad)}/{len(texts)} random texts identical")
    for t in bad[:5]:
        print("  differs:", repr(t))
    # 随机文本里混了制表符、换行这类搜索框里打不出来的字符，个别不一致可以接受；多了说明规则不对
    if len(bad) > len(texts) // 100:
        raise SystemExit("too many tokenizer mismatches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--ocr", required=True, help="prepare_ppocr.py 的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    meta = MODELS[args.model]
    model_id = meta["id"]
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="siglip2-")

    model = AutoModel.from_pretrained(meta["repo"], cache_dir=args.cache, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(meta["repo"], cache_dir=args.cache)
    tok = processor.tokenizer
    ip = processor.image_processor
    size = int(ip.size["height"]) if isinstance(ip.size, dict) else int(ip.size)
    mean = [float(v) for v in ip.image_mean]
    std = [float(v) for v in ip.image_std]
    length = int(model.config.text_config.max_position_embeddings)

    # ---- 分词器 ----
    tok_src = hf_hub_download(meta["repo"], "tokenizer.model", cache_dir=args.cache)
    tok_out = os.path.join(args.out, f"{model_id}-tokenizer.model")
    shutil.copyfile(tok_src, tok_out)
    check_tokenizer(tok, tok_out)
    probe = "a cat"
    plain = tok(probe, add_special_tokens=False)["input_ids"]
    full = tok(probe)["input_ids"]
    bos = full[0] if len(full) > len(plain) and full[1:1 + len(plain)] == plain else -1
    body = full[1:] if bos >= 0 else full
    eos = body[len(plain)] if len(body) > len(plain) else -1
    if body[:len(plain)] != plain or len(body) > len(plain) + 1:
        raise SystemExit(f"unexpected special tokens: {full} vs {plain}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def encode(texts):
        # SigLIP 2 训练时文本转了小写、补齐到固定长度
        return tok([t.lower() for t in texts], padding="max_length", max_length=length, truncation=True,
                   return_tensors="np")["input_ids"].astype(np.int64)

    tests = [{"text": t, "ids": [int(v) for v in row]} for t, row in zip(TOKENIZER_TESTS, encode(TOKENIZER_TESTS))]
    if tok.truncation_side != "right":
        raise SystemExit(f"unexpected truncation side: {tok.truncation_side}")
    print("special tokens: bos", bos, "eos", eos, "pad", pad, "length", length, "padding", tok.padding_side)

    # ---- 导出 ----
    rng = np.random.default_rng(0)
    # 低频噪声 + 少量高频细节：统计上比纯白噪声更像真实照片（已归一化的输入）
    coarse = torch.from_numpy(rng.normal(0, 0.6, (6, 3, 14, 14)).astype(np.float32))
    imgs = torch.nn.functional.interpolate(coarse, size=size, mode="bicubic", align_corners=False)
    imgs = (imgs + 0.1 * torch.randn(imgs.shape, generator=torch.Generator().manual_seed(0))).clamp(-1, 1).numpy()
    texts = ["a photo of a cat", "海边的日落", "city at night", "生日蛋糕", "a dog playing in the snow", "收据"]
    toks = encode(texts)
    with torch.no_grad():
        ref_img = model.get_image_features(pixel_values=torch.from_numpy(imgs)).numpy()
        ref_txt = model.get_text_features(input_ids=torch.from_numpy(toks)).numpy()
        logit_scale = float(model.logit_scale.exp().item())
        logit_bias = float(model.logit_bias.item())

    img_fp32 = os.path.join(tmp, "img.onnx")
    txt_fp32 = os.path.join(tmp, "txt.onnx")
    with torch.no_grad():
        # 大模型超过 2GB 时 torch 会自动把权重写成外部数据文件
        torch.onnx.export(ImageEncoder(model), (torch.zeros(1, 3, size, size),), img_fp32, input_names=["pixel_values"],
                          output_names=["embedding"], opset_version=17, do_constant_folding=True, dynamo=False)
        torch.onnx.export(TextEncoder(model), (torch.from_numpy(toks[:1]),), txt_fp32, input_names=["input_ids"],
                          output_names=["embedding"], opset_version=17, do_constant_folding=True, dynamo=False)

    def run(path, name, x):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return np.concatenate([s.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])

    for label, path, name, x, ref in [
        ("img fp32", img_fp32, "pixel_values", imgs, ref_img),
        ("txt fp32", txt_fp32, "input_ids", toks, ref_txt),
    ]:
        c = cos(run(path, name, x), ref)
        print(f"{label}: cosine vs torch min={c.min():.4f} mean={c.mean():.4f}")
        if c.min() < 0.999:
            raise SystemExit(f"{label} drifted too far from the PyTorch model ({c.min():.4f})")

    img_q = os.path.join(args.out, f"{model_id}-img.onnx")
    txt_q = os.path.join(args.out, f"{model_id}-txt.onnx")

    # 量化：从最省空间的方案往下试，第一个够准的就用；都不够准就失败（fp32 太大，手机上跑不动）
    def quantize(src, dst, name, x, ref, label):
        attempts = [
            # 连词表（Gather）一起量化：文本编码器里 256k 词表占了一大半
            ("int8 matmul+gather", dict(weight_type=QuantType.QInt8, per_channel=True,
                                        op_types_to_quantize=["MatMul", "Gemm", "Gather"])),
            ("int8 matmul-only", dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])),
            # ORT 的 ConvInteger 只支持 uint8 权重
            ("uint8 all", dict(weight_type=QuantType.QUInt8, per_channel=False)),
        ]
        for what, kw in attempts:
            try:
                quantize_dynamic(src, dst, use_external_data_format=False, **kw)
                c = cos(run(dst, name, x), ref)
            except Exception as e:  # noqa: BLE001
                print(f"{label} {what}: not usable ({e})")
                continue
            print(f"{label} {what}: cosine vs torch min={c.min():.4f} mean={c.mean():.4f} size={os.path.getsize(dst) / 1e6:.1f} MB")
            if c.min() >= 0.98:
                return
        raise SystemExit(f"{label}: no int8 variant is accurate enough")

    quantize(img_fp32, img_q, "pixel_values", imgs, ref_img, "img")
    quantize(txt_fp32, txt_q, "input_ids", toks, ref_txt, "txt")

    # 量化后文字和图片之间的相对排序应保持一致
    q_img = run(img_q, "pixel_values", imgs)
    q_txt = run(txt_q, "input_ids", toks)
    ref_rank = np.argsort(-cos(ref_txt[:, None, :], ref_img[None, :, :]), axis=1)[:, 0]
    q_rank = np.argsort(-cos(q_txt[:, None, :], q_img[None, :, :]), axis=1)[:, 0]
    print("best image per text ref/int8:", ref_rank.tolist(), q_rank.tolist())

    # ---- OCR ----
    with open(os.path.join(args.ocr, "ocr.json"), encoding="utf-8") as f:
        ocr = json.load(f)
    for f in ocr["files"]:
        src = os.path.join(args.ocr, f["name"])
        if sha256(src) != f["sha256"]:
            raise SystemExit(f"OCR 文件校验失败：{f['name']}")
        shutil.copyfile(src, os.path.join(args.out, f["name"]))

    files = []
    for p in (img_q, txt_q, tok_out):
        files.append({"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)})
        print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")
    files += ocr["files"]

    p_min = MIN_PROBABILITY
    min_score = (math.log(p_min / (1 - p_min)) - logit_bias) / logit_scale
    manifest = {
        "id": model_id,
        "version": meta["version"],
        "arch": "siglip2",
        "order": meta["order"],
        "title": meta["title"],
        "summary": meta["summary"],
        "minRamMb": meta["minRamMb"],
        "dim": int(ref_img.shape[-1]),
        "imageSize": size,
        "mean": mean,
        "std": std,
        "image": os.path.basename(img_q),
        "text": os.path.basename(txt_q),
        "tokenizer": {
            "file": os.path.basename(tok_out),
            "length": length,
            "padId": int(pad),
            "bosId": int(bos),
            "eosId": int(eos),
            "lowercase": True,
            "padLeft": tok.padding_side == "left",
            "tests": tests,
        },
        "logitScale": logit_scale,
        "logitBias": logit_bias,
        "search": {
            "minScore": round(min(0.15, max(0.0, min_score)), 4),
            "relativeCutoff": round(RELATIVE_LOGITS / logit_scale, 4),
        },
        "ocr": {k: v for k, v in ocr.items() if k not in ("files", "license")},
        "license": f"SigLIP 2 (Apache-2.0) https://huggingface.co/{meta['repo']}; {ocr['license']}",
        "files": files,
    }
    with open(os.path.join(args.out, f"manifest-{model_id}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("tokenizer", "files")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
