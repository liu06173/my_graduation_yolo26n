# ============================================================
#  YOLO26n + VisDrone2019 项目管理
#  用法: make <target>
# ============================================================

.PHONY: help setup train resume eval infer export clean status pause kill logs env-setup fetch-data start-train

help:  ## 显示帮助信息
	@echo "============================================"
	@echo " YOLO26n + VisDrone2019 命令列表"
	@echo "============================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Cloud Studio 三步启动:"
	@echo "    make env-setup   # 1. 配置环境"
	@echo "    make fetch-data  # 2. 获取数据"
	@echo "    make start-train # 3. 开始训练"
	@echo ""

setup:  ## 一键安装环境 (本地开发)
	bash setup.sh

env-setup:  ## Cloud Studio环境配置 (conda/CUDA/bypy)
	bash scripts/setup_env.sh

fetch-data:  ## 通过bypy下载VisDrone + 转换为YOLO格式
	bash scripts/fetch_data.sh

start-train:  ## 一键开始训练 (自动检测GPU)
	bash scripts/start_train.sh

train:  ## 新建训练
	bash scripts/train.sh

resume:  ## 从最近checkpoint恢复训练
	bash scripts/train.sh --resume

eval:  ## 评估模型 (mAP)
	bash scripts/eval.sh

export:  ## 导出模型为ONNX
	bash scripts/export.sh --format onnx

export-engine:  ## 导出模型为TensorRT
	bash scripts/export.sh --format engine

infer:  ## 推理 (用法: make infer SRC=image.jpg)
	bash scripts/infer.sh $(SRC)

download-data:  ## 下载VisDrone2019数据集
	bash scripts/download_data.sh

pause:  ## 暂停训练 (保存checkpoint后退出)
	python tools/train_ctrl.py pause

kill:  ## 强制终止训练
	python tools/train_ctrl.py kill

status:  ## 查看训练状态
	python tools/train_ctrl.py status

logs:  ## 查看训练日志 (最近50行)
	python tools/train_ctrl.py logs -n 50

clean:  ## 清理训练输出
	@echo "Cleaning runs/ directory..."
	@rm -rf runs/detect
	@rm -f runs/.train_pid runs/.train_status.json runs/train.log
	@echo "Done."
