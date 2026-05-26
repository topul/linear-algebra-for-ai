"""
第02课：矩阵与矩阵乘法
========================
AI 关联：
  - 神经网络的"一层"本质就是 y = Wx + b（一次矩阵乘法 + 一次加法）
  - LLM里的每个Linear层、Attention里的 Q/K/V 投影，全部是矩阵乘法
  - GPU之所以快，就是因为擅长并行算矩阵乘法
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils import zh_font

# ============================================================
# 1. 矩阵 = 一堆向量按行/列堆叠
# ============================================================

# 一个 batch 的输入：每行是一个样本（词向量）
# shape = (batch_size, embedding_dim) = (3, 4)
X = np.array([
    [0.2, 0.8, -0.1, 0.5],   # king
    [0.3, 0.7, -0.2, 0.6],   # queen
    [0.1, 0.6, -0.3, 0.4],   # man
])
print("输入 X 的形状:", X.shape)   # (3, 4) → 3个样本，每个4维

# 取行 / 列
print("第0行（king向量）:", X[0])
print("第1列:",            X[:, 1])

# ============================================================
# 2. 矩阵乘法 —— 神经网络一层的核心
# ============================================================

# 全连接层的权重矩阵 W：把4维输入投影到2维输出
# shape = (input_dim, output_dim) = (4, 2)
np.random.seed(42)
W = np.random.randn(4, 2) * 0.3
b = np.array([0.1, -0.1])   # 偏置

print("\nW 的形状:", W.shape)
print("W =\n", W)

# 矩阵乘法：Y = X @ W + b
# (3, 4) @ (4, 2) → (3, 2)
Y = X @ W + b
print("\n输出 Y 的形状:", Y.shape)
print("Y =\n", Y)

# 这就是一层全连接神经网络！
# 在PyTorch里写法就是：y = nn.Linear(4, 2)(x)

# ============================================================
# 3. 形状对齐规则 —— 维度是"管道"
# ============================================================

# 矩阵乘法 (m, k) @ (k, n) → (m, n)
# 中间的 k 必须相等！

A = np.random.randn(5, 3)
B = np.random.randn(3, 7)
C = A @ B
print(f"\n{A.shape} @ {B.shape} → {C.shape}")

# 错误示例：维度不匹配
try:
    bad = A @ np.random.randn(4, 7)   # 3 ≠ 4
except ValueError as e:
    print("形状不匹配错误:", e)

# ============================================================
# 4. 三种常见 multiply 区分
# ============================================================

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("\n逐元素乘  a * b  =\n",   a * b)        # Hadamard积
print("矩阵乘    a @ b  =\n",   a @ b)         # 矩阵乘法
print("等价写法 np.matmul =\n", np.matmul(a, b))
print("等价写法 np.dot    =\n", np.dot(a, b))   # 二维时等同 @

# ============================================================
# 5. 转置(Transpose) —— Attention 中的 K.T 就用它
# ============================================================

print("\n原矩阵 W 形状:", W.shape)         # (4, 2)
print("转置 W.T 形状:",  W.T.shape)       # (2, 4)
print("W.T =\n", W.T)

# Attention 公式里：Q @ K.T  → 把 K 的列变成行才能算分数

# ============================================================
# 6. 批量矩阵乘法 —— LLM 里 batch 维度的处理
# ============================================================

# 真实Transformer的tensor形状：(batch, seq_len, dim)
batch, seq_len, dim = 2, 4, 8
X_batch = np.random.randn(batch, seq_len, dim)
W2      = np.random.randn(dim, 16)

# numpy 自动对 batch 维度做广播
Y_batch = X_batch @ W2
print(f"\n批处理矩阵乘: {X_batch.shape} @ {W2.shape} → {Y_batch.shape}")
# (2, 4, 8) @ (8, 16) → (2, 4, 16)

# ============================================================
# 7. 可视化：矩阵乘法的几何意义
# ============================================================

# 把2D平面上的点用矩阵 R 旋转45度
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

points = np.array([[1, 0], [1, 1], [0, 1], [0.5, 0.5]])  # 一些2D点
rotated = points @ R.T

fig, ax = plt.subplots(figsize=(7, 7))
for p, label in zip(points, ['A', 'B', 'C', 'D']):
    ax.plot(*p, 'bo', markersize=10)
    ax.annotate(label, p, fontsize=12, xytext=(5, 5), textcoords='offset points')
for p, label in zip(rotated, ['A\'', 'B\'', 'C\'', 'D\'']):
    ax.plot(*p, 'rs', markersize=10)
    ax.annotate(label, p, fontsize=12, xytext=(5, 5), textcoords='offset points', color='red')

# 连线显示对应关系
for p, q in zip(points, rotated):
    ax.plot([p[0], q[0]], [p[1], q[1]], 'gray', alpha=0.3, linestyle='--')

ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlim(-1.5, 1.5);  ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.set_title("矩阵乘法 = 几何变换（这里是旋转45°）",
             fontsize=14, fontproperties=zh_font)
ax.legend(['原点(蓝)', '变换后(红)'], loc='upper left',
          prop=zh_font if zh_font else None)
plt.savefig(os.path.join(os.path.dirname(__file__), "lesson02_visualization.png"), dpi=150)
plt.close()
print("\n可视化已保存到 lesson02_visualization.png")

# ============================================================
# 8. 实战：手写一个"全连接层"类
# ============================================================

class Linear:
    """简化版的 nn.Linear"""
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)  # He初始化
        self.b = np.zeros(out_dim)

    def __call__(self, x):
        return x @ self.W + self.b


# 用我们的Linear搭一个2层网络
layer1 = Linear(4, 8)
layer2 = Linear(8, 2)

h = layer1(X)         # (3, 4) → (3, 8)
out = layer2(h)       # (3, 8) → (3, 2)
print(f"\n2层网络: {X.shape} → {h.shape} → {out.shape}")
print("最终输出:\n", out)

# ============================================================
# 练习
# ============================================================
print("\n" + "="*50)
print("练习：")
print("1. 创建 X(4×3) 和 W(3×5)，计算 X@W 并验证形状")
print("2. 写一个 3 层网络: Linear(10→20) → Linear(20→20) → Linear(20→5)")
print("3. 验证：(A@B).T == B.T @ A.T （转置的乘法律）")
