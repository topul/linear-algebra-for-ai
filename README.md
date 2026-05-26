# 线性代数 · AI 大模型学习路径

> 目标：掌握理解 Transformer / LLM 所需的全部线性代数，每一步都用 Python 实践。

## 🚀 启动 Jupyter

**强烈推荐**用 Jupyter Notebook 交互学习（点一个 cell 运行一段，比脚本直观得多）：

```bash
cd /home/cody/ai-learn
.venv/bin/jupyter lab
```

执行后会自动打开浏览器（如果没自动打开，看终端里的链接，复制粘贴）。

然后双击左边的 `01_numpy_vectors/lesson01.ipynb` 开始第一课。

### Notebook 操作速查

| 操作 | 快捷键 |
|------|-------|
| 运行当前 cell（光标停下） | `Shift + Enter` |
| 运行当前 cell（光标到下一格） | `Ctrl + Enter` |
| 在下方插入新 cell | `B` |
| 在上方插入新 cell | `A` |
| 删除当前 cell | `D` `D`（按两次D） |
| 切换 cell 类型为 Markdown | `M` |
| 切换 cell 类型为代码 | `Y` |

> 💡 修改任意 cell 的代码再 Shift+Enter，立刻看到新结果 —— **这就是 Jupyter 最大的好处，鼓励你边学边改**。

## 📚 课程地图

| 课 | 主题 | 文件 | 与 AI/LLM 的关联 |
|----|------|------|------------------|
| **01** | 向量基础 | `lesson01.ipynb` ✅ | 词嵌入、余弦相似度 |
| 02 | 矩阵与矩阵乘法 | 待写 | 全连接层、批处理 |
| 03 | 线性变换与基变换 | 待写 | 嵌入空间映射 |
| 04 | 范数、内积、正交 | 待写 | 注意力机制、归一化 |
| 05 | 秩、行列式、逆 | 待写 | 参数冗余、可逆性 |
| 06 | 特征值与特征向量 | 待写 | PCA、稳定性 |
| 07 | SVD（奇异值分解） | 待写 | **LoRA 微调** 的数学根基 |
| 08 | 矩阵微积分 | 待写 | **反向传播** |
| 09 | 投影 & 最小二乘 | 待写 | 回归、线性探针 |
| 10 | 张量与广播 | 待写 | Transformer 维度运算 |
| 11 | 综合实战：手撕注意力 | 待写 | 理解 Transformer 核心 |

## 📁 每一课的文件结构

```
01_numpy_vectors/
├── lesson01.ipynb              ← 主战场，用 Jupyter 打开
```

## 🛠 环境

- Python 3.12 + venv（在 `.venv/`）
- numpy / matplotlib / sympy / jupyter（已装好）

如果哪个 cell 报错说找不到包，确认你启动的是 `.venv/bin/jupyter lab`（不是系统 jupyter）。

## 🎯 学习节奏

1. 启动 jupyter，打开 `lesson01.ipynb`
2. **从上到下逐个 cell 运行**，边运行边读
3. 看到 TODO 的 cell 自己动手填，做不出来看下面"答案" cell
4. 改改代码玩一玩，比如把数字换掉看结果
