#!/usr/bin/env python3
"""将 VisDrone 10类标签过滤为 2类（人物+汽车），实时中文汇报进度"""
import sys, os
from pathlib import Path
from collections import Counter

# 映射规则: VisDrone原始类别 -> 新类别
# 0:pedestrian -> 0:person
# 1:people     -> 0:person
# 3:car        -> 1:car
# 其余类别(2,4,5,6,7,8,9) -> 丢弃

KEEP_MAP = {
    0: 0,   # pedestrian -> person
    1: 0,   # people -> person
    3: 1,   # car -> car
}

BASE = Path("data/visdrone")
LABELS_DIR = BASE / "labels"
NEW_LABELS_DIR = BASE / "labels_2cls"

def filter_labels(split):
    src_dir = LABELS_DIR / split
    dst_dir = NEW_LABELS_DIR / split
    if not src_dir.exists():
        print(f"  ⚠️ {split} 目录不存在，跳过")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.txt"))
    total = len(files)
    total_lines = 0
    kept_lines = 0
    cls_dist = Counter()
    empty_files = 0

    for i, f in enumerate(files):
        lines = f.read_text().strip().splitlines()
        total_lines += len(lines) if lines else 0
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id in KEEP_MAP:
                new_cls = KEEP_MAP[cls_id]
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
                cls_dist[new_cls] += 1
                kept_lines += 1

        if new_lines:
            (dst_dir / f.name).write_text("\n".join(new_lines))
        else:
            (dst_dir / f.name).write_text("")
            empty_files += 1

        # 每100张汇报
        if (i + 1) % 500 == 0 or i == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"  [{i+1}/{total}] {pct:.0f}%  |  保留: {kept_lines} 目标  |  空图: {empty_files} 张")

    return {
        "total": total,
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "empty_files": empty_files,
        "cls_dist": cls_dist,
    }

def main():
    print("=" * 55)
    print("  VisDrone 10类 → 2类 标签过滤")
    print("  人物 (pedestrian+people) + 汽车 (car)")
    print("=" * 55)
    print()

    for split in ["train", "val"]:
        print(f"📂 处理 {split} 集...")
        stats = filter_labels(split)
        if stats:
            print(f"  ✅ {split}: {stats['total']}张 -> 保留{stats['kept_lines']}个目标")
            print(f"     人物: {stats['cls_dist'][0]}个 | 汽车: {stats['cls_dist'][1]}个")
            print(f"     空图: {stats['empty_files']}张")
        print()

    # 写入新config
    config = """# YOLO26 VisDrone 2-class config (person + car)
path: ../data/visdrone
train: images/train
val: images/val
test: images/test

# 2 classes
nc: 2
names:
  0: person
  1: car
"""
    config_path = Path("configs/visdrone_2cls.yaml")
    config_path.write_text(config)
    print(f"📝 配置文件: {config_path}")

    print()
    print("=" * 55)
    print("  ✅ 过滤完成！")
    print(f"  新标签目录: {NEW_LABELS_DIR}")
    print(f"  新配置文件: {config_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
