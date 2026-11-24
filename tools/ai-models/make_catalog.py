"""
把各档模型的 manifest-<id>.json 汇总成 catalog.json，App 用它列出可选的模型。
有 addon-*.json（prepare_gemma.py 准备的最高档选配）时一并写进 catalog.json 的 addon。

用法：python make_catalog.py dist/
"""
import glob
import json
import os
import sys


def main():
    root = sys.argv[1]
    models = []
    for path in sorted(glob.glob(os.path.join(root, "manifest-*.json"))):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        models.append({
            "id": m["id"],
            "version": m["version"],
            "order": m.get("order", 0),
            "title": m.get("title", m["id"]),
            "summary": m.get("summary", ""),
            "minRamMb": m.get("minRamMb", 0),
            "bytes": sum(x["bytes"] for x in m["files"]),
            "manifest": os.path.basename(path),
        })
    if not models:
        raise SystemExit("no manifest-*.json found")
    models.sort(key=lambda x: x["order"])
    catalog = {"models": models}
    addons = sorted(glob.glob(os.path.join(root, "addon-*.json")))
    if len(addons) > 1:
        raise SystemExit(f"只支持一个选配：{addons}")
    if addons:
        with open(addons[0], encoding="utf-8") as f:
            addon = json.load(f)
        ids = {m["id"] for m in models}
        if not set(addon["tiers"]) <= ids:
            raise SystemExit(f"选配要配的档位不在列表里：{addon['tiers']}")
        catalog["addon"] = addon
    with open(os.path.join(root, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)
    print(json.dumps(models, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
