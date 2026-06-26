# 向量、矩阵与运算

> 每个神经网络都是矩阵乘法加了点别的东西。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Learning Objectives

- 实现一个 Matrix 类,支持逐元素运算、矩阵乘、转置、行列式、求逆
- 区分逐元素乘和矩阵乘,讲清楚各自什么时候用
- 只用自己写的 Matrix 类,实现一个稠密神经网络层(`relu(W @ x + b)`)
- 解释 broadcasting 规则,以及神经网络框架里 bias 是怎么加上的

## The Problem

你想建一个神经网络。读了代码看到这一行:

```
output = activation(weights @ input + bias)
```

那个 `@` 就是矩阵乘法。`weights` 是个矩阵,`input` 是个向量。如果你不知道这些运算在干嘛,这一行就是魔法。知道了,这就是某一层整个 forward pass 的三步操作。

模型处理的每张图都是像素值的矩阵,每个 word embedding 都是向量,每层神经网络都是矩阵变换。不把矩阵运算玩熟,你做不了 AI 系统,就像不懂变量写不了代码一样。

这节课从零开始,把这套熟练度建起来。

## The Concept

### 向量:有序的数字列表

向量是有方向、有大小的一组数字。在 AI 里,向量代表数据点、特征或参数。

```
v = [3, 4]        -- 一个 2D 向量
w = [1, 0, -2]    -- 一个 3D 向量
```

2D 向量 `[3, 4]` 指向平面上的 (3, 4) 点。它的长度(模)是 5(3-4-5 三角形)。

### 矩阵:数字的网格

矩阵是 2D 网格,有行有列。一个 m × n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2×3 矩阵(2 行 3 列)
    | 4  5  6 |
```

神经网络里,权重矩阵把输入向量变成输出向量。一个 784 输入、128 输出的层用 128×784 的权重矩阵。

### 为什么 shape 重要

矩阵乘法有铁律: `(m × n) @ (n × p) = (m × p)`。内维必须对得上。

```
(128 × 784) @ (784 × 1) = (128 × 1)
   权重          输入         输出

内维: 784 = 784  -- 合法
```

PyTorch 报 shape 不对,就是这个原因。

### 运算速查表

| 运算 | 干什么 | 神经网络里的用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素合并 | 把 bias 加到输出上 |
| 标量乘 | 缩放每个元素 | 学习率 × 梯度 |
| 矩阵乘 | 变换向量 | 层的前向传播 |
| 转置 | 行列互换 | 反向传播 |
| 行列式 | 一个总览数 | 检查可逆性 |
| 求逆 | 反着变换 | 解线性方程组 |
| 单位矩阵 | "啥也不干"矩阵 | 初始化、残差连接 |

### 逐元素 vs 矩阵乘

这个区别是初学者最容易栽的。

逐元素:对应位置相乘。两个矩阵必须同形。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘:行和列做点积。内维必须对得上。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

两个不同运算,两种结果,两套规则。

### Broadcasting

给一个输出矩阵加 bias 向量时,shape 对不上。Broadcasting 把小的那个拉长,直到能配上。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting 把向量沿行拉长:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都自动这么干。搞懂它,以后看到"shape 看着不对但代码能跑"就不慌了。

## Build It

### Step 1: Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: 核心运算齐备的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: 跑一下看看

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: 接到神经网络上

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个稠密层:`output = relu(W @ x + b)`。每个神经网络里的每个稠密层干的就是这个。

## Use It

NumPy 用更少的代码干同样的事,而且快上几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (逐元素) =\n", A * B)
print("A @ B (矩阵乘) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"输出:\n{output}")
```

Python 里 `@` 调的是 `__matmul__`。NumPy 用 C 和 Fortran 写的优化 BLAS 实现这个运算。数学一样,快 100 倍。

NumPy 里的 broadcasting:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动把 1D 的 bias 沿两行 broadcast。这就是每个神经网络框架里 bias 加法的工作方式。

## Ship It

本节产出 `outputs/prompt-matrix-operations.md`,一个用几何直觉教矩阵运算的 prompt。

这里写的 Matrix 类是 Phase 3 Lesson 10 那个迷你神经网络框架的底子。

## Exercises

1. **验证逆矩阵。** 算 `A @ A.inverse_2x2()` 确认得到单位矩阵。换 3 个不同的 2×2 矩阵试。行列式为 0 时会怎样?

2. **实现 3×3 求逆。** 用伴随矩阵法扩展 Matrix 类,支持 3×3 求逆。跟 NumPy 的 `np.linalg.inv` 对一下。

3. **建一个两层网络。** 只能用自己的 Matrix 类(不用 NumPy),搭一个两层网络:输入 (3) -> 隐藏 (4) -> 输出 (2)。随机初始化权重,跑 forward pass,确认所有 shape 对得上。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "一支箭头" | 一组有序的数字。AI 里:高维空间里的一个点。 |
| Matrix | "一张数字表" | 一个线性变换,把向量从一个空间映到另一个。 |
| Matrix multiply | "把数字乘起来" | 第一个矩阵的每一行和第二个矩阵的每一列做点积。顺序很关键。 |
| Transpose | "翻过来" | 行列互换。m × n 变 n × m。反向传播里离不开。 |
| Determinant | "矩阵给的一个数" | 衡量矩阵把面积(2D)或体积(3D)放大了多少。0 意味着这个变换把某一维压扁了。 |
| Inverse | "把矩阵撤回去" | 能反着变换的矩阵。只在行列式不为 0 时存在。 |
| Identity matrix | "无聊的矩阵" | 相当于乘以 1 的矩阵版本。残差连接(ResNet)里要用。 |
| Broadcasting | "魔法 shape 修复" | 把小数组沿缺的维重复,撑到跟大数组匹配。 |
| Element-wise | "普通乘法" | 对应位置相乘。两个数组要么同形,要么能 broadcast。 |

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) —— 本节每个运算的可视化直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) —— NumPy 遵循的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) —— ML 专用线性代数速查
