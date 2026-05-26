"""
第01课 练习答案
做完 exercises 之后再来看，对一下答案。
"""

import numpy as np

# === 题1：余弦相似度函数 ===
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king  = np.array([0.2, 0.8, -0.1, 0.5])
queen = np.array([0.3, 0.7, -0.2, 0.6])
cat   = np.array([0.5, 0.9, 0.1, 0.7])
dog   = np.array([0.4, 0.85, 0.15, 0.65])

print(f"king-queen: {cosine_sim(king, queen):.4f}")
print(f"cat-dog:    {cosine_sim(cat, dog):.4f}")
# cat-dog 接近 1，因为这两个向量我设的几乎平行


# === 题2：学习率 ===
old_params = np.array([10.0, 20.0, 30.0])
gradient   = np.array([5.0, -3.0, 2.0])

for lr in [0.001, 0.1, 1.0, 10.0]:
    new_params = old_params - lr * gradient   # ← 关键
    print(f"lr={lr:>6}: {new_params}")


# === 题3：垂直向量 ===
A = np.array([3, 4, 0])
B = np.array([0, 0, 5])

dot_product = np.dot(A, B)          # = 3*0 + 4*0 + 0*5 = 0
similarity  = cosine_sim(A, B)      # = 0 / (5 × 5) = 0
print(f"A·B={dot_product}, cos={similarity}")
# 结论：两个向量垂直 ⇔ 点积为 0
# 这是线性代数最有用的事实之一，记住它！
