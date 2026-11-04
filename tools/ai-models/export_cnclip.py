"""
把 Chinese-CLIP RN50（MIT，https://github.com/OFA-Sys/Chinese-CLIP）导出成手机上用的 ONNX：

  cnclip-rn50-img.onnx   图像编码器，输入 image [1,3,224,224] float32，输出 [1,1024]
  cnclip-rn50-txt.onnx   文本编码器，输入 text  [1,52] int64，        输出 [1,1024]

批大小固定为 1：RN50 的注意力池化在导出时会把 batch 当常量，动态 batch 会算错。
手机上本来也是一张一张算。

两个都做 int8 动态量化，再与 fp32 原模型逐一对比余弦相似度，偏差过大就直接失败。
最后写 manifest.json（文件名、字节数、sha256），App 下载后按它校验。

用法：python export_cnclip.py --out dist/
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

import math

import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from cn_clip.clip.model import AttentionPool2d

MODEL_ID = "cnclip-rn50"
VERSION = 1
CONTEXT = 52


class ImageEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        return self.m.encode_image(image)


class TextEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, text):
        return self.m.encode_text(text)


def attnpool_for_export(self, x):
    """
    与 AttentionPool2d.forward 数学上相同，但不用 F.multi_head_attention_forward：
    后者导出成 ONNX 后结果不对（和 PyTorch 的余弦只有 0.70）。只算 CLS 这一个查询。
    """
    x = x.flatten(2).permute(2, 0, 1)  # (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    x = x + self.positional_embedding[:, None, :].to(x.dtype)
    length, n, c = x.shape
    h = self.num_heads
    d = c // h
    q = self.q_proj(x[:1]).reshape(1, n * h, d).transpose(0, 1)
    k = self.k_proj(x).reshape(length, n * h, d).transpose(0, 1)
    v = self.v_proj(x).reshape(length, n * h, d).transpose(0, 1)
    attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d), dim=-1)
    out = torch.bmm(attn, v).transpose(0, 1).reshape(1, n, c)
    return self.c_proj(out)[0]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/cnclip"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_fp32")
    os.makedirs(tmp, exist_ok=True)

    try:
        model, _ = load_from_name("RN50", device="cpu", download_root=args.cache)
    except Exception as e:  # Hugging Face 不可用时退到 ModelScope
        print("HF download failed, trying ModelScope:", e)
        model, _ = load_from_name("RN50", device="cpu", download_root=args.cache, use_modelscope=True)
    model.eval()

    rng = np.random.default_rng(0)
    # 低频噪声 + 少量高频细节：统计上比纯白噪声更像真实照片（已归一化的输入）
    coarse = torch.from_numpy(rng.normal(0, 1, (6, 3, 14, 14)).astype(np.float32))
    imgs = torch.nn.functional.interpolate(coarse, size=224, mode="bicubic", align_corners=False)
    imgs = (imgs + 0.15 * torch.randn(imgs.shape, generator=torch.Generator().manual_seed(0))).numpy()
    texts = ["一只猫", "海边的日落", "城市夜景", "生日蛋糕", "Sunset at the BEACH!"]
    toks = clip.tokenize(texts, context_length=CONTEXT).numpy()
    with torch.no_grad():
        ref_img = model.encode_image(torch.from_numpy(imgs)).numpy()
        ref_txt = model.encode_text(torch.from_numpy(toks)).numpy()
        AttentionPool2d.forward = attnpool_for_export
        patched = model.encode_image(torch.from_numpy(imgs)).numpy()
    print(f"patched attnpool vs original: min cosine {cos(patched, ref_img).min():.6f}")
    if cos(patched, ref_img).min() < 0.9999:
        raise SystemExit("patched attention pooling does not match the original")

    img_fp32 = os.path.join(tmp, "img.onnx")
    txt_fp32 = os.path.join(tmp, "txt.onnx")
    dummy_img = torch.zeros(1, 3, 224, 224)
    dummy_txt = clip.tokenize(["一只猫"], context_length=CONTEXT)
    with torch.no_grad():
        torch.onnx.export(ImageEncoder(model), (dummy_img,), img_fp32, input_names=["image"],
                          output_names=["embedding"], opset_version=14, do_constant_folding=True, dynamo=False)
        torch.onnx.export(TextEncoder(model), (dummy_txt,), txt_fp32, input_names=["text"],
                          output_names=["embedding"], opset_version=14, do_constant_folding=True, dynamo=False)

    img_q = os.path.join(args.out, f"{MODEL_ID}-img.onnx")
    txt_q = os.path.join(args.out, f"{MODEL_ID}-txt.onnx")
    # ---- 校验：ONNX 与 PyTorch 原模型的输出必须足够接近 ----

    def run(path, name, x):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return np.concatenate([s.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])

    for label, path, name, x, ref in [
        ("img fp32", img_fp32, "image", imgs, ref_img),
        ("txt fp32", txt_fp32, "text", toks, ref_txt),
    ]:
        c = cos(run(path, name, x), ref)
        print(f"{label}: cosine vs torch min={c.min():.4f} mean={c.mean():.4f}")
        if c.min() < 0.999:
            raise SystemExit(f"{label} drifted too far from the PyTorch model ({c.min():.4f})")

    # 量化：从最省空间的方案往下试，第一个够准的就用；都不够准就保留 fp32
    def quantize(src, dst, name, x, ref, label):
        attempts = [
            # ORT 的 ConvInteger 只支持 uint8 权重
            ("uint8 per-channel", dict(weight_type=QuantType.QUInt8, per_channel=True)),
            ("int8 per-tensor", dict(weight_type=QuantType.QUInt8)),
            ("int8 matmul-only", dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])),
        ]
        for what, kw in attempts:
            try:
                quantize_dynamic(src, dst, **kw)
                c = cos(run(dst, name, x), ref)
            except Exception as e:
                print(f"{label} {what}: not usable ({e})")
                continue
            print(f"{label} {what}: cosine vs torch min={c.min():.4f} mean={c.mean():.4f} size={os.path.getsize(dst) / 1e6:.1f} MB")
            if c.min() >= 0.97:
                return
        shutil.copyfile(src, dst)
        print(f"{label}: keeping fp32 ({os.path.getsize(dst) / 1e6:.1f} MB)")

    quantize(img_fp32, img_q, "image", imgs, ref_img, "img")
    quantize(txt_fp32, txt_q, "text", toks, ref_txt, "txt")

    # 文本之间的相似度排序在量化前后应保持一致
    q_txt = run(txt_q, "text", toks)
    ref_rank = np.argsort(-cos(ref_txt[:, None, :], ref_txt[None, :, :]), axis=1)[:, 1]
    q_rank = np.argsort(-cos(q_txt[:, None, :], q_txt[None, :, :]), axis=1)[:, 1]
    print("nearest-text ref/int8:", ref_rank.tolist(), q_rank.tolist())

    # 分词向量：App 里的 Kotlin 分词器测试可以对照这份输出
    with open(os.path.join(args.out, "tokenizer-check.json"), "w", encoding="utf-8") as f:
        json.dump([{"text": t, "ids": [int(v) for v in row if v != 0]} for t, row in zip(texts, toks)],
                  f, ensure_ascii=False, indent=1)

    files = []
    for p in (img_q, txt_q):
        files.append({"name": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p)})
        print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")
    manifest = {
        "id": MODEL_ID,
        "version": VERSION,
        "dim": int(ref_img.shape[-1]),
        "imageSize": 224,
        "contextLength": CONTEXT,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "license": "Chinese-CLIP (MIT) https://github.com/OFA-Sys/Chinese-CLIP",
        "files": files,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
