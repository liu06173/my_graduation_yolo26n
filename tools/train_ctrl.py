#!/usr/bin/env python3
"""
训练管理工具
用法:
  python tools/train_ctrl.py pause     # 优雅暂停训练 (保存checkpoint后退出)
  python tools/train_ctrl.py status    # 查看训练状态
  python tools/train_ctrl.py kill      # 强制终止训练
  python tools/train_ctrl.py logs -n 50 # 查看最近50行训练日志
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import json
from pathlib import Path

PID_FILE = "runs/.train_pid"
LOG_FILE = "runs/train.log"
STATUS_FILE = "runs/.train_status.json"


def get_status():
    """Get current training status."""
    status = {
        "running": False,
        "pid": None,
        "epoch": "unknown",
        "gpu_mem": "unknown",
        "last_update": None,
    }

    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            status["running"] = True
            status["pid"] = pid
        except OSError:
            status["running"] = False

    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            saved = json.load(f)
            status.update(saved)

    return status


def cmd_pause():
    """Pause training gracefully - send SIGINT to trigger checkpoint save."""
    status = get_status()
    if not status["running"]:
        print("No training process is running.")
        return

    pid = status["pid"]
    print(f"Sending pause signal to PID {pid}...")
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGINT)
        print("Pause signal sent. Training will save checkpoint and exit.")
        print("Resume with: bash scripts/train.sh --resume")
    except Exception as e:
        print(f"Failed to send signal: {e}")


def cmd_status():
    """Display training status."""
    status = get_status()
    print("=" * 50)
    print(" Training Status")
    print("=" * 50)
    print(f"  Running:     {'YES' if status['running'] else 'NO'}")
    print(f"  PID:         {status.get('pid', 'N/A')}")
    print(f"  Epoch:       {status.get('epoch', 'N/A')}")
    print(f"  GPU Memory:  {status.get('gpu_mem', 'N/A')}")
    print(f"  Last Update: {status.get('last_update', 'N/A')}")
    print("=" * 50)

    # Check for latest weights
    for pattern in ["runs/detect/train/weights/best.pt",
                    "runs/detect/train/weights/last.pt",
                    "runs/detect/train2/weights/best.pt"]:
        if os.path.exists(pattern):
            size = os.path.getsize(pattern) / 1024 / 1024
            mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(os.path.getmtime(pattern)))
            print(f"\n  Latest weights: {pattern} ({size:.1f}MB, {mtime})")
            break


def cmd_logs(n=50):
    """Display recent training logs."""
    log_file = None
    for pattern in [LOG_FILE,
                    "runs/detect/train/results.csv",
                    "runs/detect/train2/results.csv"]:
        if os.path.exists(pattern):
            log_file = pattern
            break

    if log_file:
        with open(log_file) as f:
            lines = f.readlines()
        print(f"--- {log_file} (last {min(n, len(lines))} lines) ---")
        for line in lines[-n:]:
            print(line.rstrip())
    else:
        print("No log file found.")


def cmd_kill():
    """Force kill training."""
    status = get_status()
    if not status["running"]:
        print("No training process is running.")
        return

    pid = status["pid"]
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Training process (PID {pid}) killed.")
    except Exception as e:
        print(f"Failed to kill: {e}")

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def main():
    parser = argparse.ArgumentParser(description="Training Control Tool")
    sub = parser.add_subparsers(dest="command", help="Commands")

    sub.add_parser("pause", help="Pause training gracefully")
    sub.add_parser("status", help="Show training status")
    sub.add_parser("kill", help="Force kill training")
    logs_p = sub.add_parser("logs", help="Show recent training logs")
    logs_p.add_argument("-n", type=int, default=50, help="Number of lines")

    args = parser.parse_args()

    if args.command == "pause":
        cmd_pause()
    elif args.command == "status":
        cmd_status()
    elif args.command == "kill":
        cmd_kill()
    elif args.command == "logs":
        cmd_logs(args.n)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
