#!/usr/bin/env python3
"""多目标检测演示 — 实时中文汇报结果"""

import sys, os, time, json
from pathlib import Path
from collections import Counter

import cv2
import torch
from ultralytics import YOLO

# ---------- 中英文类别映射 ----------
COCO_CN = {
    "person": "行人", "bicycle": "自行车", "car": "轿车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停止标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "滑雪板",
    "sports ball": "运动球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子",
    "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽",
    "refrigerator": "冰箱", "book": "书", "clock": "钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}

# 无人机场景重点关注类别
FOCUS_CLASSES = {"car", "truck", "bus", "person", "bicycle", "motorcycle", "van"}


def fmt_box(cls_name, conf, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    cn = COCO_CN.get(cls_name, cls_name)
    return f"  [{cn}] 置信度:{conf:.2f}  框:[{x1},{y1},{x2},{y2}]  {x2-x1}x{y2-y1}"


def main():
    # ---------- 加载模型 ----------
    print("=" * 55)
    print("  🚁 YOLO26n 无人机航拍多目标检测")
    print("=" * 55)
    print(f"  PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("-" * 55)

    t0 = time.time()
    print("⏳ 正在加载预训练模型 yolo26n.pt ...")
    model = YOLO("yolo26n.pt")
    print(f"✅ 模型加载完成，耗时 {time.time()-t0:.1f}s")
    print(f"   类别数: {len(model.names)} 类")

    # ---------- 查找待检测图片 ----------
    data_dir = Path("data/visdrone/images/val")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        sys.exit(1)

    images = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    print(f"📁 检测目录: {data_dir}")
    print(f"🖼️  图片数量: {len(images)} 张")
    print("=" * 55)
    print()

    if not images:
        print("❌ 没有找到图片")
        sys.exit(1)

    # ---------- 逐张检测 ----------
    total = len(images)
    all_counts = Counter()
    proc_times = []

    for idx, img_path in enumerate(images, 1):
        t_start = time.time()

        # 读取图像尺寸
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # 推理
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device=0,
            verbose=False,
        )

        elapsed = time.time() - t_start
        proc_times.append(elapsed)

        result = results[0]
        boxes = result.boxes
        n_det = len(boxes) if boxes is not None else 0

        # 统计类别
        img_counts = Counter()
        if boxes is not None and n_det > 0:
            for cls_id in boxes.cls:
                cls_name = model.names[int(cls_id)]
                img_counts[cls_name] += 1
            all_counts.update(img_counts)

        # ---------- 实时中文汇报 ----------
        print(f"[{idx}/{total}] 📷 {img_path.name}  ({w}x{h})  —  检出 {n_det} 个目标  ⏱️ {elapsed*1000:.0f}ms")
        if n_det > 0:
            # 按数量降序排列
            for cls_name, cnt in img_counts.most_common():
                cn = COCO_CN.get(cls_name, cls_name)
                marker = "⚠️ " if cls_name in FOCUS_CLASSES else "  "
                print(f"  {marker}{cn}: {cnt}个")
        else:
            print("  (无检出目标)")
        print()

    # ---------- 汇总报告 ----------
    print()
    print("=" * 55)
    print("  📊 最终检测汇总报告")
    print("=" * 55)
    print(f"  处理图片: {total} 张")
    print(f"  总目标数: {sum(all_counts.values())} 个")
    avg_time = sum(proc_times) / len(proc_times) * 1000 if proc_times else 0
    print(f"  平均耗时: {avg_time:.0f}ms/张")
    print(f"  总耗时:   {sum(proc_times):.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU显存:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print("-" * 55)
    print("  各类别统计 (按数量降序):")
    print()
    for cls_name, cnt in all_counts.most_common():
        cn = COCO_CN.get(cls_name, cls_name)
        bar = "█" * min(cnt, 40)
        print(f"  {cn:<8} {cnt:>5}个  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    main()
