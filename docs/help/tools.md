# 工具使用问题

---

## bypy — 百度网盘命令行客户端

### 当前状态
- 版本: v1.8.9
- 配额: 12.020TB / 已用: 6.039TB
- 可用空间: ~6TB

### UTF-8警告

**现象**:
```
WARNING: System locale is not 'UTF-8'. Current locale is '936'
```

**解决**: 在使用bypy命令前临时设置编码:
```powershell
$env:PYTHONUTF8=1
$env:PYTHONIOENCODING='utf-8'
bypy info
```

或永久修改：Windows设置 → 时间和语言 → 语言和区域 → 管理语言设置 → 更改系统区域设置 → 勾选 "Beta: 使用Unicode UTF-8提供全球语言支持" → 重启。

### 常用命令

```bash
# 查看配额
bypy info

# 列出云盘根目录
bypy list

# 列出指定目录
bypy list /datasets

# 下载文件
bypy downfile /datasets/VisDrone2019.zip ./data/

# 下载整个目录 (递归)
bypy downdir /datasets/VisDrone2019-DET ./data/visdrone/

# 上传文件
bypy upload ./my_model.pt /models/

# 上传整个目录
bypy upload ./runs/ /训练结果/

# 同步下载 (增量，只下载变化的文件)
bypy syncdown /datasets/VisDrone2019 ./data/visdrone/

# 同步上传
bypy syncup ./runs/ /训练结果/

# 查看当前用户
bypy whoami
```

### 下载VisDrone2019的思路

1. 找VisDrone2019的百度网盘分享链接
2. 保存到自己的百度网盘
3. 用bypy下载:
```bash
# 假设保存在 /VisDrone2019 目录下
bypy downdir /VisDrone2019/VisDrone2019-DET-train ./data/visdrone/VisDrone2019-DET-train/
bypy downdir /VisDrone2019/VisDrone2019-DET-val ./data/visdrone/VisDrone2019-DET-val/
```

### 加速下载
```bash
# 多进程 + aria2下载器
bypy downdir /VisDrone2019 ./data/visdrone/ --processes 4 --downloader aria2 --select-fastest-mirror
```
