#!/usr/bin/env bash
# 推送到 GitHub 的脚本
# 用法：bash push_to_github.sh
#
# 前置：
#   - gh 已安装并 login 过（gh auth status 能看到账号）
#   - 代理可用（如果需要）：先 export 代理变量

set -e

REPO_NAME="linear-algebra-for-ai"
VISIBILITY="--public"          # 改成 --private 可以私有
DESCRIPTION="线性代数学习笔记 · 面向 AI 大模型 (LLM/Transformer)，含 Jupyter notebook 和 Python 实践"

# 如果你需要代理，取消下面三行的注释（地址改成你的代理）
# export https_proxy=http://172.20.32.1:7890
# export http_proxy=http://172.20.32.1:7890
# export all_proxy=socks5://172.20.32.1:7890

cd "$(dirname "$0")"

echo "==> 检查 gh 认证状态"
gh auth status

echo "==> 创建 GitHub 远程仓库 $REPO_NAME"
# --source=. 把当前本地仓库关联过去
# --push 自动推送 main 分支
gh repo create "$REPO_NAME" \
    $VISIBILITY \
    --description "$DESCRIPTION" \
    --source=. \
    --remote=origin \
    --push

echo "==> 完成！"
gh repo view --web
