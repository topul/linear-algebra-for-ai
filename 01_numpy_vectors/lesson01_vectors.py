"""
================================================================
第01课：向量 —— AI的"语言"
================================================================
建议配合 README.md 一起看。
代码分8段，每段对应一个小知识点，请逐段运行、看输出、读注释。

运行方式：
    cd /home/cody/ai-learn
    .venv/bin/python 01_numpy_vectors/lesson01_vectors.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils import zh_font


def section(title):
    """打印分节标题，让输出更清晰"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
section("【第1段】什么是向量？怎么在Python里创建？")
# ============================================================
#
# 在 Python 中，我们用 numpy 库来表示向量。
# numpy 是科学计算的标准库，几乎所有 AI 框架（PyTorch、TensorFlow）
# 底层的数学运算都参考它的设计。
#
# 习惯上，import numpy as np 是行业标准写法。
# ------------------------------------------------------------

# 创建一个向量：用 np.array() 把一个 Python 列表包起来
xiaoming = np.array([175, 70, 25, 8000])
print("小明这个人 =", xiaoming)
#
# 输出: [175   70   25 8000]
# 这就是一个 4 维向量，4 个数字分别代表：身高、体重、年龄、收入
#

# 看几个属性
print("向量里有几个数字？", xiaoming.shape)
# 输出: (4,)
# 这个 (4,) 的意思是：1维数组，长度4。
# 后面的逗号是 numpy 表示"一维"的固定写法，先不用纠结。

print("数据类型是？", xiaoming.dtype)
# 输出: int64 （整数）
# 如果你给的数字里有小数，numpy 会自动用 float64（浮点数）

# 注意：直接用 Python 列表 [175, 70, ...] 也能存数据，
# 但 Python 列表做数学运算非常慢，而且没有"加法/乘法"的向量含义。
# numpy 数组才是为数学量身定做的。


# ============================================================
section("【第2段】AI里的向量：词嵌入(Word Embedding)")
# ============================================================
#
# 真实 AI 里，"国王" 这个词会被表示成 300 维（甚至更高）的向量，
# 每个数字代表这个词的某种"特征"。这些特征是机器自己学出来的，
# 通常不太能直接解释。
#
# 为了方便理解，我们用 4 维向量模拟：
# ------------------------------------------------------------

king  = np.array([0.2, 0.8, -0.1, 0.5])   # "国王" 的向量
queen = np.array([0.3, 0.7, -0.2, 0.6])   # "王后" 的向量
man   = np.array([0.1, 0.6, -0.3, 0.4])   # "男人" 的向量

print("king  =", king)
print("queen =", queen)
print("man   =", man)

# 这些数字现在是我编出来的，但概念上，真实的词向量就是这样
# —— 每个词都是一串浮点数。


# ============================================================
section("【第3段】向量加法：对应位置相加")
# ============================================================
#
# 规则：向量1的第i个 + 向量2的第i个 = 结果的第i个
# 要求：两个向量维度必须相同
# ------------------------------------------------------------

a = np.array([3, 0])  # 往东走3步
b = np.array([0, 4])  # 往北走4步

total = a + b
print(f"{a} + {b} = {total}")
# 输出: [3 0] + [0 4] = [3 4]
# 几何意义：先东3再北4，相当于直接走到 (3, 4) 这个点

# 同样可以减
print(f"{a} - {b} = {a - b}")
# 输出: [3 0] - [0 4] = [3 -4]


# AI 里的实际场景：参数更新
# ---------------------------
# 神经网络训练时，每一步都要更新参数：
#   新参数 = 旧参数 - 学习率 × 梯度
#
# 这里"参数"和"梯度"都是巨大的向量（GPT-3里有1750亿个数字！）
# 但运算规则就是上面这条：对应位置相减。

old_params = np.array([1.0, 2.0, 3.0])
gradient   = np.array([0.5, -0.2, 0.1])
lr = 0.1   # 学习率，控制每次走多大步

new_params = old_params - lr * gradient
print(f"参数更新示例: {old_params} → {new_params}")


# ============================================================
section("【第4段】标量乘法：拉长/缩短向量")
# ============================================================
#
# 一个数字 × 一个向量 = 每个分量都乘那个数字
# ------------------------------------------------------------

v = np.array([1, 2, 4])

print(f"2 × {v} = {2 * v}")        # 拉长2倍
print(f"0.5 × {v} = {0.5 * v}")    # 缩到一半
print(f"-1 × {v} = {-1 * v}")      # 方向反过来


# ============================================================
section("【第5段】点积（关键！）：衡量两个向量的'一致性'")
# ============================================================
#
# 计算规则：对应位置相乘，全部加起来。结果是一个数字。
#
#   [a₁, a₂, a₃] · [b₁, b₂, b₃] = a₁b₁ + a₂b₂ + a₃b₃
#
# 含义：方向越一致，点积越大；方向越相反，点积越小（甚至负）
# ------------------------------------------------------------

# 手动算一遍，验证你的理解
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 方式1：用 numpy 的函数
dot1 = np.dot(A, B)
print(f"np.dot({A}, {B}) = {dot1}")

# 方式2：用 @ 符号（Python 3.5+ 专门为矩阵乘法引入的）
dot2 = A @ B
print(f"{A} @ {B} = {dot2}")

# 方式3：手动算一遍（验证）
manual = 1*4 + 2*5 + 3*6
print(f"手动: 1×4 + 2×5 + 3×6 = {manual}")


# 直观例子：向量方向越一致，点积越大
# -----------------------------------
right     = np.array([1, 0])         # 朝右
right2    = np.array([2, 0])         # 也朝右（更长）
up        = np.array([0, 1])         # 朝上（垂直）
left      = np.array([-1, 0])        # 朝左（相反）

print(f"\n方向一致的点积: right · right2 = {right @ right2}")     # 正数大
print(f"方向垂直的点积: right · up     = {right @ up}")          # 0
print(f"方向相反的点积: right · left   = {right @ left}")        # 负数


# 在 AI 中：点积 = 注意力分数
# -------------------------------
# 当模型处理 "猫吃鱼" 时：
print("\n模拟注意力打分:")
fish = np.array([0.3, 0.9, 0.1])   # 鱼的向量
cat  = np.array([0.4, 0.85, 0.2])  # 猫的向量
eat  = np.array([0.2, 0.7, 0.4])   # 吃的向量
de   = np.array([0.01, 0.02, 0.9]) # 的的向量（虚词）

print(f"  鱼·猫  = {fish @ cat:.3f}  ← 相关性高")
print(f"  鱼·吃  = {fish @ eat:.3f}  ← 相关性高")
print(f"  鱼·的  = {fish @ de:.3f}  ← 相关性低（'的'是虚词）")


# ============================================================
section("【第6段】向量的长度（L2 范数）")
# ============================================================
#
# 公式：‖v‖ = √(v₁² + v₂² + ... + vₙ²)
# 就是把勾股定理推广到任意维度。
# ------------------------------------------------------------

v = np.array([3, 4])

# 手动算
manual_length = np.sqrt(3**2 + 4**2)
print(f"手动: √(3² + 4²) = √25 = {manual_length}")

# 用 numpy 的函数（实际开发都这么写）
length = np.linalg.norm(v)
print(f"np.linalg.norm({v}) = {length}")
# linalg = linear algebra（线性代数）模块
# norm   = 范数（向量长度的专业说法）

# 高维向量长度也是一样的规则
v3d = np.array([1, 2, 2])
print(f"‖{v3d}‖ = {np.linalg.norm(v3d)}")   # √9 = 3


# 归一化：让向量长度变成 1
# --------------------------
# 公式：v / ‖v‖
# 用途：注意力机制里的"normalize"步骤、BERT/GPT 的归一化层
v = np.array([3, 4])
v_unit = v / np.linalg.norm(v)
print(f"\n{v} 归一化后 = {v_unit}")
print(f"归一化后的长度 = {np.linalg.norm(v_unit):.6f}  (≈1.0)")


# ============================================================
section("【第7段】余弦相似度（重头戏）")
# ============================================================
#
# 公式：cos(θ) = (A · B) / (‖A‖ × ‖B‖)
#
# 这是 AI 里衡量"两个东西像不像"的标准方法。
# 值的范围：[-1, 1]
#   1   = 完全一样的方向（最像）
#   0   = 互相垂直（无关）
#   -1  = 完全相反的方向（最不像）
# ------------------------------------------------------------

def cosine_sim(a, b):
    """
    余弦相似度：忽略向量长度，只看方向是否一致
    """
    dot_product = np.dot(a, b)
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    return dot_product / (length_a * length_b)


# 测试1：方向相同但长度不同 → 相似度应该是 1
v1 = np.array([1, 1])
v2 = np.array([100, 100])  # 跟 v1 同方向，但长度差100倍
print(f"cos({v1}, {v2}) = {cosine_sim(v1, v2):.3f}  ← 同方向，无视长度")

# 测试2：垂直 → 应该是 0
v3 = np.array([1, 0])
v4 = np.array([0, 1])
print(f"cos({v3}, {v4}) = {cosine_sim(v3, v4):.3f}  ← 垂直")

# 测试3：相反 → 应该是 -1
v5 = np.array([1, 1])
v6 = np.array([-1, -1])
print(f"cos({v5}, {v6}) = {cosine_sim(v5, v6):.3f}  ← 完全相反")


# 实际应用：词向量的相似度
print("\n词向量相似度测试:")
print(f"  cos(king, queen) = {cosine_sim(king, queen):.3f}  ← 高，因为都是'王室'")
print(f"  cos(king, man)   = {cosine_sim(king, man):.3f}  ← 略低，都是'男性'相关")


# ============================================================
section("【第8段】可视化")
# ============================================================
# 把上面讲的概念画出来，加深直觉
# ------------------------------------------------------------

words_2d = {
    "king":  np.array([0.8, 0.6]),
    "queen": np.array([0.7, 0.7]),
    "man":   np.array([0.3, 0.4]),
    "woman": np.array([0.4, 0.5]),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：4个词向量的方向
ax = axes[0]
colors = ['blue', 'red', 'green', 'purple']
for (word, vec), color in zip(words_2d.items(), colors):
    ax.annotate("",
                xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color=color))
    ax.text(vec[0]*1.08, vec[1]*1.08, word, fontsize=13, color=color)
ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.set_title("4个词在2D空间中的方向", fontsize=14, fontproperties=zh_font)
ax.grid(alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)

# 右图：相似度矩阵（颜色越深 = 越相似）
ax = axes[1]
words = list(words_2d.keys())
n = len(words)
sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim[i, j] = cosine_sim(words_2d[words[i]], words_2d[words[j]])

im = ax.imshow(sim, cmap="YlOrRd", vmin=0.85, vmax=1.0)
ax.set_xticks(range(n)); ax.set_xticklabels(words)
ax.set_yticks(range(n)); ax.set_yticklabels(words)
ax.set_title("两两余弦相似度（越红越像）", fontsize=14, fontproperties=zh_font)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=12)
plt.colorbar(im, ax=ax)

plt.tight_layout()
output = os.path.join(os.path.dirname(__file__), "lesson01_visualization.png")
plt.savefig(output, dpi=120)
plt.close()
print(f"\n✅ 可视化图已生成: {output}")


# ============================================================
section("练习时间")
# ============================================================
print("""
请打开 lesson01_exercises.py，完成 3 道题：

题1：动物词向量
  cat = [0.5, 0.9, 0.1, 0.7]
  dog = [0.4, 0.85, 0.15, 0.65]
  - 算 cat 和 dog 的余弦相似度
  - 跟 king-queen 的相似度比较，谁更"像"？

题2：学习率的影响
  把 lr 改成 0.001、0.1、1.0，观察参数更新幅度有什么变化？
  从直觉上想：学习率太大会发生什么？太小呢？

题3：手动验证
  对 A = [3, 4, 0] 和 B = [0, 0, 5]：
  - 它们的点积应该是多少？
  - 它们应该是垂直还是平行？为什么？
  - 用代码验证你的答案
""")
