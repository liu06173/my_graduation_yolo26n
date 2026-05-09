#!/usr/bin/env python3
"""MOT格式 → YOLO格式转换 + 2类过滤 (person + vehicle)
VisDrone2019-MOT → YOLO person(0) + vehicle(1)
实时中文汇报进度"""
import os, sys
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# MOT类别 → 新类别映射
# MOT: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
#       5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus,
#       10=motor, 11=others
MOT_TO_NEW = {
    1: 0,   # pedestrian → person
    2: 0,   # people → person
    4: 1,   # car → vehicle
    5: 1,   # van → vehicle
    6: 1,   # truck → vehicle
    9: 1,   # bus → vehicle
    # 0,3,7,8,10,11 → 丢弃
}

CN_NAME = {0: "人物(person)", 1: "车辆(vehicle)"}

DATA_ROOT = Path("data/VisDrone2019-MOT-train/VisDrone2019-MOT-train")
VAL_ROOT = Path("data/VisDrone2019-MOT-val/VisDrone2019-MOT-val")
OUT_DIR = Path("D:/yolo26_cache/data/visdrone")

def get_image_dims(img_path):
    """从图片获取尺寸 (带缓存)"""
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except:
        return None, None

def convert_mot_to_yolo(mot_root, out_split, split_name):
    """转换一个split的MOT数据"""
    ann_dir = mot_root / "annotations"
    seq_dir = mot_root / "sequences"

    if not ann_dir.exists():
        print(f"  ⚠️ {split_name}: 标注目录不存在 {ann_dir}")
        return None

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"  ⚠️ {split_name}: 无标注文件")
        return None

    out_img_dir = OUT_DIR / "images" / out_split
    out_lbl_dir = OUT_DIR / "labels" / out_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_targets = 0
    kept_targets = 0
    cls_dist = Counter()
    skipped_no_img = 0
    skipped_no_dim = 0
    empty_frames = 0

    for ann_file in ann_files:
        seq_name = ann_file.stem  # e.g. "uav0000013_00000_v"
        seq_path = seq_dir / seq_name

        if not seq_path.exists():
            print(f"  ⚠️ 序列目录不存在: {seq_name}")
            continue

        # 读取所有标注行，按frame分组
        lines = ann_file.read_text().strip().splitlines()
        frame_anns = defaultdict(list)
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            frame_anns[frame_idx].append(parts)

        # 处理每一帧
        for frame_idx, anns in frame_anns.items():
            total_frames += 1
            frame_name = f"{seq_name}_{frame_idx:07d}"
            src_img = seq_path / f"{frame_idx:07d}.jpg"
            dst_img = out_img_dir / f"{frame_name}.jpg"
            dst_lbl = out_lbl_dir / f"{frame_name}.txt"

            # 获取图片尺寸
            img_w, img_h = get_image_dims(src_img)
            if img_w is None or img_h is None or img_w == 0:
                skipped_no_dim += 1
                continue

            yolo_lines = []
            for parts in anns:
                total_targets += 1
                mot_cls = int(parts[7])
                if mot_cls not in MOT_TO_NEW:
                    continue
                new_cls = MOT_TO_NEW[mot_cls]
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                # 归一化YOLO格式: <class> <x_center> <y_center> <width> <height>
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # 边界裁剪
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                if norm_w <= 0 or norm_h <= 0:
                    continue

                yolo_lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                cls_dist[new_cls] += 1
                kept_targets += 1

            # 写标签文件
            if yolo_lines:
                dst_lbl.write_text("\n".join(yolo_lines))
            else:
                dst_lbl.write_text("")
                empty_frames += 1

            # 复制图片 (用硬链接或符号链接节省空间)
            if not dst_img.exists() and src_img.exists():
                try:
                    os.link(str(src_img), str(dst_img))
                except OSError:
                    import shutil
                    shutil.copy2(str(src_img), str(dst_img))

        # 每处理一个序列汇报
        print(f"  [{seq_name}] 帧数:{len(frame_anns):4d}  |  累计保留:{kept_targets:7d}  |  空帧:{empty_frames}")

    return {
        "total_frames": total_frames,
        "total_targets": total_targets,
        "kept_targets": kept_targets,
        "empty_frames": empty_frames,
        "cls_dist": cls_dist,
        "skipped_no_dim": skipped_no_dim,
    }

def main():
    print("=" * 60)
    print("  VisDrone2019-MOT → YOLO 2类转换")
    print("  人物(person) + 车辆(vehicle: car/van/truck/bus)")
    print("=" * 60)
    print()

    results = {}

    # 训练集
    print("📂 处理训练集 (VisDrone2019-MOT-train)...")
    results["train"] = convert_mot_to_yolo(DATA_ROOT, "train", "train")
    if results["train"]:
        r = results["train"]
        print(f"  ✅ 训练集: {r['total_frames']}张有效帧, "
              f"保留{r['kept_targets']}/{r['total_targets']}个目标 "
              f"(丢弃{r['total_targets']-r['kept_targets']}个)")
        for cls_id, name in CN_NAME.items():
            print(f"     {name}: {r['cls_dist'].get(cls_id, 0)}个")
        if r["empty_frames"]:
            print(f"     空帧(无目标): {r['empty_frames']}张")

    print()

    # 验证集
    print("📂 处理验证集 (VisDrone2019-MOT-val)...")
    if VAL_ROOT.exists():
        results["val"] = convert_mot_to_yolo(VAL_ROOT, "val", "val")
        if results["val"]:
            r = results["val"]
            print(f"  ✅ 验证集: {r['total_frames']}张有效帧, "
                  f"保留{r['kept_targets']}/{r['total_targets']}个目标")
            for cls_id, name in CN_NAME.items():
                print(f"     {name}: {r['cls_dist'].get(cls_id, 0)}个")
    else:
        print(f"  ⚠️ 验证集目录不存在: {VAL_ROOT}")

    # 写配置文件
    config = f"""# YOLO26 VisDrone 2-class config (person + vehicle)
# 从VisDrone2019-MOT转换而来
# 类别: 0=person(行人+人群), 1=vehicle(轿车+面包车+卡车+公交)
path: D:/yolo26_cache/data/visdrone
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: person
  1: vehicle
"""
    config_path = Path("configs/visdrone_2cls.yaml")
    config_path.write_text(config)
    print(f"\n📝 配置文件更新: {config_path}")
    print(f"   nc=2, names: 0=person, 1=vehicle")

    print()
    print("=" * 60)
    print("  ✅ 转换完成！")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  配置文件: {config_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
