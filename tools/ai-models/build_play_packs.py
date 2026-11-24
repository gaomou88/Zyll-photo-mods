"""
把 Release 里下载好的各档模型分装成 Google Play 按需下载的资源包。

用法：python build_play_packs.py <下载目录> <android 目录> [修图模型下载目录]

- 每档模型一个或几个资源包（Play 限制单个按需资源包不超过 1.5 GB，超出就按文件拆开）
- 资源包模块生成在 android/ai-packs/<包名>/，settings.gradle.kts 和 app 会自动带上
- App 用的模型列表写到 android/app/src/play/assets/ai/catalog.json：
  每档的信息、完整的 manifest，以及它在哪几个资源包里
- 最高档的 Gemma 选配（catalog.json 的 addon）：已经切好的每一段一个资源包，放在 assets/llm/，
  写进 catalog.json 的 addon（每段多一个 pack 字段）
- 修图模型每个一个资源包，列表写到 android/app/src/play/assets/ai/editor-catalog.json（多一个 pack 字段）
"""
import hashlib
import json
import os
import re
import shutil
import sys

# Play 的上限是 1.5 GB，留一点余量
PACK_LIMIT = 1400 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_gradle(name):
    return (
        "// 由 tools/ai-models/build_play_packs.py 生成，不要手改\n"
        "plugins {\n"
        '    id("com.android.asset-pack")\n'
        "}\n\n"
        "assetPack {\n"
        f'    packName.set("{name}")\n'
        "    dynamicDelivery {\n"
        '        deliveryType.set("on-demand")\n'
        "    }\n"
        "}\n"
    )


def main():
    src, android = sys.argv[1], sys.argv[2]
    packs_root = os.path.join(android, "ai-packs")
    shutil.rmtree(packs_root, ignore_errors=True)
    os.makedirs(packs_root)

    with open(os.path.join(src, "catalog.json"), encoding="utf-8") as f:
        full = json.load(f)
    catalog = full["models"]

    out = []
    total = 0
    for entry in catalog:
        with open(os.path.join(src, entry["manifest"]), encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest["id"] != entry["id"]:
            raise SystemExit(f"{entry['manifest']} 的 id 和 catalog.json 不一致")
        safe = re.sub(r"[^a-z0-9]+", "_", entry["id"].lower()).strip("_")

        # 按文件装箱，每个包不超过 PACK_LIMIT
        groups = []
        for f in manifest["files"]:
            if f["bytes"] > PACK_LIMIT:
                raise SystemExit(f"{f['name']} 有 {f['bytes']} 字节，超过单个资源包上限，需要换半精度模型")
            if not groups or sum(x["bytes"] for x in groups[-1]) + f["bytes"] > PACK_LIMIT:
                groups.append([])
            groups[-1].append(f)

        packs = []
        for i, files in enumerate(groups):
            name = f"ai_{safe}" if len(groups) == 1 else f"ai_{safe}_{i + 1}"
            mod = os.path.join(packs_root, name)
            assets = os.path.join(mod, "src", "main", "assets", "clip")
            os.makedirs(assets)
            with open(os.path.join(mod, "build.gradle.kts"), "w", encoding="utf-8") as g:
                g.write(pack_gradle(name))
            for f in files:
                path = os.path.join(src, f["name"])
                if sha256(path) != f["sha256"]:
                    raise SystemExit(f"校验失败：{f['name']}")
                # 文字识别模型每档都有一份（同名同内容），不能移走，复制
                shutil.copyfile(path, os.path.join(assets, f["name"]))
            size = sum(x["bytes"] for x in files)
            total += size
            print(f"{name}: {', '.join(x['name'] for x in files)}  {size / 1024 / 1024:.0f} MB")
            packs.append(name)

        item = {k: entry[k] for k in ("id", "version", "order", "title", "summary", "minRamMb", "bytes") if k in entry}
        item["manifest"] = manifest
        item["packs"] = packs
        out.append(item)

    result = {"models": out}
    addon = full.get("addon")
    if addon:
        result["addon"] = build_addon(src, packs_root, addon)

    assets_dir = os.path.join(android, "app", "src", "play", "assets", "ai")
    os.makedirs(assets_dir, exist_ok=True)
    with open(os.path.join(assets_dir, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"共 {len(out)} 档搜索模型，资源包合计 {total / 1024 / 1024:.0f} MB")

    if len(sys.argv) > 3:
        build_editor(sys.argv[3], packs_root, assets_dir)


def build_addon(src, packs_root, addon):
    """选配模型：每一段一个资源包，App 下载完按顺序拼回去。"""
    safe = re.sub(r"[^a-z0-9]+", "_", addon["id"].lower()).strip("_")
    parts = []
    whole = hashlib.sha256()
    for i, p in enumerate(addon["parts"]):
        if p["bytes"] > PACK_LIMIT:
            raise SystemExit(f"选配的 {p['name']} 超过单个资源包上限")
        name = f"ai_addon_{safe}_{i + 1}"
        mod = os.path.join(packs_root, name)
        assets = os.path.join(mod, "src", "main", "assets", "llm")
        os.makedirs(assets)
        with open(os.path.join(mod, "build.gradle.kts"), "w", encoding="utf-8") as g:
            g.write(pack_gradle(name))
        path = os.path.join(src, p["name"])
        if sha256(path) != p["sha256"]:
            raise SystemExit(f"校验失败：{p['name']}")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                whole.update(chunk)
        shutil.move(path, os.path.join(assets, p["name"]))
        parts.append({**p, "pack": name})
        print(f"{name}: {p['name']}  {p['bytes'] / 1024 / 1024:.0f} MB")
    if whole.hexdigest() != addon["sha256"]:
        raise SystemExit("选配各段拼起来的 sha256 对不上")
    return {k: v for k, v in addon.items() if k != "parts"} | {"parts": parts}


def build_editor(src, packs_root, assets_dir):
    with open(os.path.join(src, "editor-catalog.json"), encoding="utf-8") as f:
        catalog = json.load(f)
    for m in catalog["models"]:
        if m["size"] > PACK_LIMIT:
            raise SystemExit(f"修图模型 {m['id']} 超过单个资源包上限")
        name = "ed_" + re.sub(r"[^a-z0-9]+", "_", m["id"].lower()).strip("_")
        mod = os.path.join(packs_root, name)
        assets = os.path.join(mod, "src", "main", "assets", "editor")
        os.makedirs(assets)
        with open(os.path.join(mod, "build.gradle.kts"), "w", encoding="utf-8") as g:
            g.write(pack_gradle(name))
        path = os.path.join(src, m["file"])
        if sha256(path) != m["sha256"].lower():
            raise SystemExit(f"校验失败：{m['file']}")
        shutil.move(path, os.path.join(assets, m["file"]))
        m["pack"] = name
        print(f"{name}: {m['file']}  {m['size'] / 1024 / 1024:.0f} MB")
    with open(os.path.join(assets_dir, "editor-catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
