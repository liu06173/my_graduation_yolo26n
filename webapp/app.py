#!/usr/bin/env python3
"""YOLO26 视频检测对比平台 — Flask Web App"""
import sys, os, json, shutil, uuid, time, subprocess, threading, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import torch
import cv2

# ─── 日志系统 ───
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m-%d %H:%M:%S')

file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger('webapp')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=" * 50)
logger.info("WebApp starting")

# ─── 路径配置 ───
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
UPLOADS_DIR = BASE / "uploads"
OUTPUTS_DIR = BASE / "outputs"
VIDEO_DIR = BASE / "videos"  # 用户放置视频的文件夹
STATIC_DIR = BASE / "static"

for d in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ─── 全局状态 ───
active_tasks = {}
model_cache = {}

# ─── 模型扫描 ───
def scan_models():
    """扫描models目录下所有非last.pt模型"""
    models = {}
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        pt_files = [f for f in model_dir.glob("*.pt") if f.stem != "last"]
        if pt_files:
            models[model_dir.name] = {
                "path": str(model_dir),
                "weights": [{"name": f.name, "path": str(f), "size_mb": round(f.stat().st_size / 1e6, 2)} for f in sorted(pt_files)],
                "count": len(pt_files),
            }
    return models

def scan_videos():
    """扫描videos目录和uploads目录"""
    videos = []
    for d in [VIDEO_DIR, UPLOADS_DIR]:
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}:
                    videos.append({
                        "name": f.name,
                        "path": str(f),
                        "folder": d.name,
                        "size_mb": round(f.stat().st_size / 1e6, 2),
                    })
    return videos

# ─── 检测引擎 ───
class DetectionEngine:
    def __init__(self):
        self.models = {}

    def _smart_device(self):
        """自动选择设备: GPU显存>3GB空闲用GPU, 否则CPU"""
        if torch.cuda.is_available():
            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
            if free_mem > 3.0:
                logger.info(f"[DEVICE] GPU free={free_mem:.1f}GB, using CUDA")
                return 0
            else:
                logger.warning(f"[DEVICE] GPU free={free_mem:.1f}GB < 3GB, falling back to CPU")
        return 'cpu'

    def load_model(self, model_path):
        if model_path not in self.models:
            from ultralytics import YOLO
            self.models[model_path] = YOLO(model_path)
        return self.models[model_path]

    def _create_video_writer(self, frame, output_path):
        """创建视频写入器，自动尝试多个编码器"""
        h, w = frame.shape[:2]
        for codec in ['mp4v', 'XVID', 'MJPG', 'DIVX']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))
            if writer.isOpened():
                logger.info(f"[VIDEO] codec={codec}, path={output_path}")
                return writer
        logger.error(f"[VIDEO] all codecs failed for {output_path}")
        return None

    def run_tracking(self, model_path, video_path, conf=0.25, iou=0.7, img_size=640):
        """多目标跟踪统计"""
        device = self._smart_device()
        logger.info(f"[TRACK] start: {Path(model_path).name} on {Path(video_path).name}, device={device}")

        model = self.load_model(str(model_path))
        results = model.track(
            source=str(video_path),
            conf=conf, iou=iou, imgsz=img_size,
            stream=True, verbose=False, persist=True,
            tracker='botsort.yaml',
            device=device,
        )
        total_objects = 0
        class_counts = defaultdict(int)
        total_infer_time = 0
        frame_count = 0
        seen_ids = defaultdict(set)

        output_name = f"track_{Path(model_path).stem}_{Path(video_path).stem}_{uuid.uuid4().hex[:6]}.mp4"
        output_video_path = OUTPUTS_DIR / output_name
        video_writer = None

        for r in results:
            frame_count += 1
            boxes = r.boxes
            plotted = r.plot()
            if video_writer is None:
                video_writer = self._create_video_writer(plotted, output_video_path)
            if video_writer:
                video_writer.write(plotted)

            if boxes is not None and len(boxes) > 0:
                n = len(boxes)
                total_objects += n
                cls_ids = boxes.cls.tolist() if hasattr(boxes, 'cls') else []
                track_ids = boxes.id.tolist() if boxes.id is not None else []
                for i, cid in enumerate(cls_ids):
                    name = model.names.get(int(cid), str(int(cid)))
                    class_counts[name] += 1
                    if i < len(track_ids) and track_ids[i] is not None:
                        seen_ids[name].add(int(track_ids[i]))
                speed = r.speed if hasattr(r, 'speed') else {}
                total_infer_time += speed.get('inference', 0) if speed else 0

        if video_writer:
            video_writer.release()

        elapsed = total_infer_time / 1000 if total_infer_time else 0
        unique_counts = {k: len(v) for k, v in seen_ids.items()}
        logger.info(f"[TRACK] done: {frame_count} frames, {int(total_objects)} detections, {sum(unique_counts.values())} unique")

        return {
            "model": Path(model_path).parent.name,
            "weight": Path(model_path).name,
            "mode": "tracking",
            "total_frames": frame_count,
            "total_detections": int(total_objects),
            "unique_objects": int(sum(unique_counts.values())),
            "unique_by_class": unique_counts,
            "avg_per_frame": round(total_objects / frame_count, 2) if frame_count else 0,
            "class_distribution": dict(class_counts),
            "infer_time_ms": round(total_infer_time / frame_count, 2) if frame_count else 0,
            "fps": round(frame_count / (total_infer_time / 1000), 1) if total_infer_time else 0,
            "conf_threshold": conf, "iou_threshold": iou,
            "output_video": str(output_video_path),
            "output_video_name": Path(output_video_path).name,
        }

    def run_detection(self, model_path, video_path, conf=0.25, iou=0.7, img_size=640):
        """运行检测，返回逐帧结果 + 生成标注视频"""
        device = self._smart_device()
        logger.info(f"[DETECT] start: {Path(model_path).name} on {Path(video_path).name}, device={device}")

        model = self.load_model(str(model_path))
        results = model.predict(
            source=str(video_path),
            conf=conf, iou=iou, imgsz=img_size,
            stream=True, verbose=False,
            device=device,
        )
        frames_data = []
        total_objects = 0
        class_counts = defaultdict(int)
        total_infer_time = 0
        frame_count = 0
        output_name = f"det_{Path(model_path).stem}_{Path(video_path).stem}_{uuid.uuid4().hex[:6]}.mp4"
        output_video_path = OUTPUTS_DIR / output_name
        video_writer = None

        for r in results:
            frame_count += 1
            boxes = r.boxes
            plotted = r.plot()
            if video_writer is None:
                video_writer = self._create_video_writer(plotted, output_video_path)
            if video_writer:
                video_writer.write(plotted)

            if boxes is not None:
                n = len(boxes)
                total_objects += n
                cls_ids = boxes.cls.tolist() if len(boxes) > 0 else []
                for cid in cls_ids:
                    name = model.names.get(int(cid), str(int(cid)))
                    class_counts[name] += 1
                speed = r.speed if hasattr(r, 'speed') else {}
                total_infer_time += speed.get('inference', 0) if speed else 0
                frames_data.append({"frame": frame_count, "objects": n,
                    "classes": {model.names.get(int(c), str(int(c))): cls_ids.count(c) for c in set(cls_ids)}})

        if video_writer:
            video_writer.release()

        elapsed = total_infer_time / 1000 if total_infer_time else 0
        logger.info(f"[DETECT] done: {frame_count} frames, {int(total_objects)} objects")

        return {
            "model": Path(model_path).parent.name,
            "weight": Path(model_path).name,
            "total_frames": frame_count,
            "total_objects": int(total_objects),
            "avg_per_frame": round(total_objects / frame_count, 2) if frame_count else 0,
            "class_distribution": dict(class_counts),
            "infer_time_ms": round(total_infer_time / frame_count, 2) if frame_count else 0,
            "fps": round(frame_count / (total_infer_time / 1000), 1) if total_infer_time else 0,
            "conf_threshold": conf, "iou_threshold": iou,
            "frames_data": frames_data[:100],
            "output_video": str(output_video_path),
            "output_video_name": Path(output_video_path).name,
        }

engine = DetectionEngine()

# ─── 路由 ───
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    return jsonify(scan_models())

@app.route('/api/videos')
def get_videos():
    return jsonify(scan_videos())

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "未选择文件"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    original = file.filename
    stem = Path(original).stem
    suffix = Path(original).suffix
    safe_name = secure_filename(stem) + suffix
    fname = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    save_path = UPLOADS_DIR / fname
    file.save(str(save_path))
    return jsonify({"name": fname, "path": str(save_path), "size_mb": round(save_path.stat().st_size / 1e6, 2)})

@app.route('/api/detect', methods=['POST'])
def run_detect():
    data = request.json
    video_path = data.get('video_path')
    model_paths = data.get('model_paths', [])
    conf = data.get('conf', 0.25)
    iou = data.get('iou', 0.7)
    img_size = data.get('img_size', 640)
    mode = data.get('mode', 'detect')

    logger.info(f"[API] detect request: mode={mode}, models={len(model_paths)}, video={video_path}")

    if not video_path or not Path(video_path).exists():
        logger.error(f"[API] video not found: {video_path}")
        return jsonify({"error": "视频文件不存在"}), 400
    if not model_paths:
        logger.error("[API] no models selected")
        return jsonify({"error": "未选择模型"}), 400

    task_id = uuid.uuid4().hex[:8]
    active_tasks[task_id] = {"status": "running", "progress": 0, "results": []}
    logger.info(f"[TASK:{task_id}] created, mode={mode}, {len(model_paths)} models")

    def _run():
        results = []
        for i, mp in enumerate(model_paths):
            pct = int(((i + 1) / len(model_paths)) * 90)
            active_tasks[task_id]["progress"] = pct
            logger.info(f"[TASK:{task_id}] [{i+1}/{len(model_paths)}] processing {mp} (progress={pct}%)")
            try:
                t0 = time.time()
                if mode == 'track':
                    r = engine.run_tracking(mp, video_path, conf, iou, img_size)
                else:
                    r = engine.run_detection(mp, video_path, conf, iou, img_size)
                elapsed = time.time() - t0
                logger.info(f"[TASK:{task_id}] [{i+1}/{len(model_paths)}] done in {elapsed:.1f}s: "
                           f"objects={r.get('total_objects', r.get('total_detections', '?'))}, "
                           f"fps={r.get('fps', '?')}, "
                           f"output_video={r.get('output_video_name', 'NONE')}, "
                           f"error={r.get('error', 'none')}")
                results.append(r)
            except Exception as e:
                import traceback
                logger.error(f"[TASK:{task_id}] [{i+1}/{len(model_paths)}] FAILED: {e}")
                logger.error(traceback.format_exc())
                results.append({"model": mp, "weight": Path(mp).name, "error": str(e)})
        active_tasks[task_id] = {
            "status": "done",
            "progress": 100,
            "results": results,
            "task_id": task_id,
        }
        logger.info(f"[TASK:{task_id}] COMPLETED: {len(results)} results, "
                   f"errors={sum(1 for r in results if r.get('error'))}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": task_id, "status": "running"})

@app.route('/api/logs')
def get_logs():
    """查看最近日志"""
    log_file = LOG_DIR / "app.log"
    if not log_file.exists():
        return jsonify({"logs": ["日志文件不存在"]})
    lines = log_file.read_text(encoding='utf-8').splitlines()
    return jsonify({"logs": lines[-100], "total": len(lines)})

@app.route('/api/task/<task_id>')
def get_task(task_id):
    task = active_tasks.get(task_id, {"status": "not_found"})
    return jsonify(task)

@app.route('/api/deepseek', methods=['POST'])
def deepseek_analyze():
    """调用DeepSeek API进行分析"""
    data = request.json
    api_key = data.get('api_key', '')
    model_name = data.get('model', 'deepseek-v4-flash')
    detection_results = data.get('results', [])
    prompt = data.get('prompt', '')

    if not api_key:
        return jsonify({"error": "请填写DeepSeek API Key"}), 400
    if not detection_results:
        return jsonify({"error": "无检测结果"}), 400

    system_prompt = """你是一个计算机视觉专家。请分析以下YOLO模型在视频上的检测结果。

对每个模型分析：
1. 检测性能（目标数、类别分布）
2. 可能的问题和改进建议
3. 如果是多模型对比，指出各模型优劣

请用中文回答，专业且简洁。"""

    user_prompt = f"检测结果对比:\n{json.dumps(detection_results, ensure_ascii=False, indent=2)}\n\n用户问题: {prompt}" if prompt else f"检测结果:\n{json.dumps(detection_results, ensure_ascii=False, indent=2)}\n\n请分析这些检测结果。"

    try:
        import requests
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return jsonify(resp.json()["choices"][0]["message"])
        else:
            return jsonify({"error": f"API Error {resp.status_code}: {resp.text[:200]}"}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """提供视频文件"""
    for d in [VIDEO_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
        fpath = d / filename
        if fpath.exists():
            return send_from_directory(str(d), filename)
    return "Not found", 404

@app.route('/output/<path:filename>')
def serve_output(filename):
    """提供检测输出视频"""
    fpath = OUTPUTS_DIR / filename
    if fpath.exists():
        return send_from_directory(str(OUTPUTS_DIR), filename)
    return "Not found", 404

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """多模型详细对比"""
    data = request.json
    results = data.get('results', [])

    if len(results) < 2:
        return jsonify({"error": "需要至少2个模型才能对比"}), 400

    # 计算对比指标
    comparison = {
        "models": [r.get("weight", r.get("model", "unknown")) for r in results],
        "metrics": {},
        "winner": {},
    }

    # 各项指标
    for key, label, higher_better in [
        ("total_objects", "检测总数", None),
        ("avg_per_frame", "每帧平均", None),
        ("fps", "推理FPS", True),
        ("infer_time_ms", "推理耗时(ms)", False),
    ]:
        vals = []
        for r in results:
            v = r.get(key, 0)
            try:
                vals.append(float(v))
            except:
                vals.append(0)
        comparison["metrics"][key] = {
            "label": label,
            "values": vals,
            "unit": "fps" if "fps" in key else "ms" if "ms" in key else "",
        }
        if higher_better is not None and len(vals) >= 2:
            idx = vals.index(max(vals)) if higher_better else vals.index(min(vals))
            comparison["winner"][key] = comparison["models"][idx]

    # 类别分布对比
    all_classes = set()
    for r in results:
        all_classes.update(r.get("class_distribution", {}).keys())
    class_comp = {}
    for cls in sorted(all_classes):
        class_comp[cls] = [r.get("class_distribution", {}).get(cls, 0) for r in results]
    comparison["class_comparison"] = class_comp

    return jsonify(comparison)

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  YOLO26 视频检测对比平台")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  视频目录: {VIDEO_DIR}  (请将视频放入此文件夹)")
    print(f"  启动地址: http://localhost:5800")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5800, debug=False, threaded=True)
