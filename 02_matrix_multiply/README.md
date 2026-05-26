# 第02课：矩阵与矩阵乘法

## 核心概念

### 1. 矩阵的本质
矩阵 = 数据的二维表格，AI里的语义：
- 一批样本：`X.shape = (batch_size, feature_dim)`，每行是一个样本
- 权重矩阵：`W.shape = (in_dim, out_dim)`，存储一层网络的全部参数

### 2. 矩阵乘法规则

```
(m, k) @ (k, n) → (m, n)
         ↑↑↑
       中间维度必须相同！
```

**算法**：`C[i,j] = Σ_k A[i,k] * B[k,j]`

直觉：C 的第 i 行第 j 列 = A 的第 i 行 与 B 的第 j 列 的点积。

### 3. 全连接层 (Linear Layer)

神经网络最基础的一层：
```
Y = X @ W + b
```
- X: 输入，形状 `(batch, in_dim)`
- W: 权重，形状 `(in_dim, out_dim)` —— 这就是"参数"
- b: 偏置，形状 `(out_dim,)`
- Y: 输出，形状 `(batch, out_dim)`

GPT-3 有 175B 参数，绝大部分就是这些 W 矩阵的元素。

### 4. 三种"乘法"区分（重要！）

| 操作 | 写法 | 形状要求 | 用途 |
|------|------|----------|------|
| 逐元素乘 (Hadamard) | `A * B` | 完全相同 | 门控、mask |
| 矩阵乘 | `A @ B` | (m,k) (k,n) | 神经网络层 |
| 点积 (vector) | `np.dot(a,b)` | (n,) (n,) | 标量结果 |

### 5. 关键性质

- **不满足交换律**：`A @ B ≠ B @ A`
- **结合律成立**：`(A @ B) @ C = A @ (B @ C)`
- **转置乘法律**：`(A @ B).T = B.T @ A.T`

### 6. 在 LLM 中的角色

Transformer 一个 block 里的矩阵乘法（不算attention细节）：
```python
x = x @ W_attn_qkv     # QKV 投影
x = x @ W_attn_proj    # 注意力输出投影
x = x @ W_ffn_up       # FFN 升维
x = x @ W_ffn_down     # FFN 降维
```
全部都是矩阵乘法。下一课会展开 Attention 内部的细节。

## 运行
```bash
.venv/bin/python 02_matrix_multiply/lesson02_matmul.py
```
