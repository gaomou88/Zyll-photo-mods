"""
准备最高档的选配：Gemma 4 E2B（Apache 2.0，https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm），
LiteRT-LM 格式，App 夜里用它给照片写描述、搜索时理解长句子。

模型 2.6 GB：GitHub Release 单个文件上限 2 GB，Play 单个资源包上限 1.5 GB，所以切成几段上传，
App 下载完按顺序拼回一个文件、校验整份的 SHA-256。

输出到 --out：
  gemma4-e2b.litertlm.part1 ...  各段
  addon-gemma4-e2b.json          选配信息：整份大小和 sha256、各段的大小和 sha256、能配哪几档
make_catalog.py 会把它并进 catalog.json。

用法：python prepare_gemma.py --out dist/
"""
import argparse
import hashlib
import json
import os

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "litert-community/gemma-4-E2B-it-litert-lm"
# 手机 CPU/GPU 用的那份（同一个仓库里还有网页版和高通 NPU 版）
FILE = "gemma-4-E2B-it.litertlm"
ADDON_ID = "gemma4-e2b"
# 照片描述按这个 key 存；换了模型或 App 里的提示语就改它，夜里会重新写
KEY = "gemma4-e2b@1"
# 只给最高档配
TIERS = ["siglip2-so400m"]
MIN_RAM_MB = 12 * 1024
# 每段不超过这么大（Play 资源包上限 1.5 GB，留余量）
PART = 1300 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = list_repo_files(REPO)
    if FILE not in files:
        raise SystemExit(f"{REPO} 里没有 {FILE}：{[f for f in files if f.endswith('.litertlm')]}")
    src = hf_hub_download(REPO, FILE, cache_dir=args.cache)
    total = os.path.getsize(src)
    whole = sha256(src)
    print(f"{FILE}: {total / 1e9:.2f} GB sha256={whole}")

    parts = []
    with open(src, "rb") as f:
        i = 0
        while True:
            i += 1
            name = f"{ADDON_ID}.litertlm.part{i}"
            path = os.path.join(args.out, name)
            h = hashlib.sha256()
            n = 0
            with open(path, "wb") as out:
                while n < PART:
                    chunk = f.read(min(1 << 20, PART - n))
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if n == 0:
                os.remove(path)
                break
            parts.append({"name": name, "bytes": n, "sha256": h.hexdigest()})
            print(f"{name}: {n / 1e6:.0f} MB")
    if sum(p["bytes"] for p in parts) != total:
        raise SystemExit("切分后大小对不上")

    # 拼回去再算一遍，确认切分没问题
    h = hashlib.sha256()
    for p in parts:
        with open(os.path.join(args.out, p["name"]), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    if h.hexdigest() != whole:
        raise SystemExit("拼回去的 sha256 对不上")

    info = {
        "id": ADDON_ID,
        "key": KEY,
        "version": 1,
        "title": "Gemma 4 E2B",
        "bytes": total,
        "sha256": whole,
        "tiers": TIERS,
        "minRamMb": MIN_RAM_MB,
        "license": f"Gemma 4 (Apache-2.0) https://huggingface.co/{REPO}",
        "parts": parts,
    }
    with open(os.path.join(args.out, f"addon-{ADDON_ID}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "parts"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
