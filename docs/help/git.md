# Git 常见问题与使用指南

---

## <a id="001"></a>001 — SSH连接GitHub超时 (Connection timed out)

**现象**:
```bash
ssh -T git@github.com
ssh: connect to host github.com port 22: Connection timed out
```

**原因**: 网络环境（如中国大陆）封锁了22端口，无法直接通过SSH协议直连GitHub。

**解决方法（3种，按推荐度排序）**:

### 方法1：SSH走443端口（推荐）

GitHub提供了 `ssh.github.com` 作为SSH over HTTPS的入口（443端口通常不会被封）。

```powershell
# 先确保目录存在
mkdir "$env:USERPROFILE\.ssh"

# 创建/编辑SSH配置文件
notepad "$env:USERPROFILE\.ssh\config"
```

粘贴以下内容:
```
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

保存后验证:
```bash
ssh -T git@github.com
```

看到 `Hi <用户名>! You've successfully authenticated...` 即成功。

### 方法2：HTTPS + Token

**Step 1 — 创建Token**: 打开 https://github.com/settings/tokens
- 点击 "Generate new token (classic)"
- 勾选 `repo` 全部权限
- 生成后**立即复制** `ghp_xxxx...`（只显示一次）

**Step 2 — 切换远程地址**:
```bash
git remote set-url origin https://github.com/你的用户名/仓库名.git
```

**Step 3 — 推送时输入**:
- Username: 你的GitHub用户名
- Password: 粘贴Token

**Step 4 — 记住密码（免重复输入）**:
```bash
git config --global credential.helper wincred
```

### 方法3：GitHub CLI
```bash
winget install GitHub.cli
gh auth login
# 选 GitHub.com → HTTPS → Login with a web browser
gh repo push
```

---

## <a id="002"></a>002 — Windows下~/.ssh目录不存在

**现象**:
```powershell
notepad ~/.ssh/config
# 系统找不到指定的路径。
```

**原因**: 从未生成过SSH密钥，`.ssh`目录不存在。

**解决**:
```powershell
mkdir ~/.ssh
# 或使用完整路径:
mkdir "$env:USERPROFILE\.ssh"
```

---

## <a id="003"></a>003 — Git配置速查

### 基础配置
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
git config --global credential.helper wincred  # Windows记住密码
```

### 生成SSH密钥
```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 一路回车即可

# 查看公钥（复制到GitHub Settings → SSH Keys）
cat ~/.ssh/id_ed25519.pub
```

### 日常操作
```bash
git status               # 查看改了哪些文件
git add -A               # 暂存所有改动
git commit -m "提交信息"  # 创建提交
git push                 # 推送到远程
git pull                 # 拉取远程更新
git log --oneline -10    # 查看最近10条提交
```

### 撤销操作
```bash
# 撤销尚未commit的修改（⚠ 不可恢复）
git checkout -- 文件名

# 撤销 git add（取消暂存）
git reset HEAD 文件名

# 回退到上一次提交（保留修改）
git reset --soft HEAD~1

# 创建新分支
git checkout -b 分支名

# 切换分支
git checkout 分支名
```

### 常见场景
```bash
# 忘记加文件了，追加到上一次commit
git add 忘记的文件
git commit --amend --no-edit

# 想重写commit信息
git commit --amend -m "新的提交信息"

# 远程分支已被删除，同步本地
git fetch --prune

# 查看某个文件的修改历史
git log -p -- 文件路径
```
