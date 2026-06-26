# 数值稳定性

> 浮点是个有漏的抽象。它会在训练中咬你一口,而且你根本看不见。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- 用"减去最大值"的技巧实现数值稳定的 softmax 和 log-sum-exp
- 在浮点计算里识别 overflow、underflow、灾难性 cancellation
- 用中心有限差分验证解析梯度
- 解释为什么 bfloat16 比 float16 更适合训练,以及 loss scaling 怎么防止梯度下溢

## The Problem

你的模型训了三个小时,loss 突然变 NaN。你加了 print。step 9000 时 logits 还正常,step 9001 就 `inf` 了。step 9002 所有梯度变 `nan`,训练死了。

或者:你的模型跑完训练,准确率比论文低 2%。你把架构、超参、数据都对了一遍。问题就是论文用 float32,你用 float16 又没加正确的缩放。32 bit 累积的舍入误差悄悄吃了你的准确率。

或者:你从零实现交叉熵 loss。logits 小的时候跑得好。logits 超过 100 就返回 `inf`。softmax 溢出了,因为 `exp(100)` 比 float32 能表示的最大数还大。每个 ML 框架都用两行 trick 处理这个。你不知道有这招。

数值稳定性不是理论问题。它是训练跑成功还是默默失败的区别。你调的每个正经 ML bug 最后都回到浮点。

## The Concept

### IEEE 754:计算机怎么存实数

计算机按 IEEE 754 标准把实数存为浮点值。一个 float 有三部分: 一个符号位、一个指数、一个尾数(有效位)。

```
Float32 布局(32 bit):
[1 符号] [8 指数] [23 尾数]

值 = (-1)^符号 * 2^(指数-127) * 1.尾数
```

尾数决定精度(多少位有效数字)。指数决定范围(数能多大或多小)。

```
格式       bit    指数     尾数     十进制数字  范围(约)
float64    64     11       52       ~15-16     +/- 1.8e308
float32    32     8        23       ~7-8       +/- 3.4e38
float16    16     5        10       ~3-4       +/- 65,504
bfloat16   16     8        7        ~2-3       +/- 3.4e38
```

float32 给大约 7 位十进制精度。它能分清 1.0000001 和 1.0000002,但分不清 1.00000001 和 1.00000002。7 位以后全是舍入噪声。

float16 给大约 3 位。它能表示的最大数是 65,504。对 ML 来说这个数小得吓人 —— logits、梯度、激活值动不动就超过。

bfloat16 是 Google 对 float16 范围不够的解法。它有跟 float32 一样的 8 bit 指数(范围一样,到 3.4e38),但只有 7 bit 尾数(比 float16 精度低)。训神经网络时,范围比精度更重要,所以 bfloat16 通常胜出。

### 为啥 0.1 + 0.2 != 0.3

0.1 这个数在二进制浮点里没法精确表示。在 base 2 里它是个无限循环小数:

```
0.1 二进制 = 0.0001100110011001100110011... (永远循环)
```

Float32 把这个截到 23 bit 尾数。存进去的值约等于 0.100000001490116。类似地,0.2 存为约 0.200000002980232。它们加起来是 0.300000004470348,不是 0.3。

```
在 Python 里:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这跟 ML 有关,因为:
1. 像 `if loss < threshold` 的 loss 比较可能给出错的答案
2. 累加很多小值(几千步的梯度更新)会跟真实总和漂移
3. 校验和跟可复现性测试如果用 `==` 比浮点会失败

修法: 永远别用 `==` 比浮点。用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性 Cancellation

当你减两个几乎相等的浮点数,有效位互相抵消,留下来的是被提升到前导位的舍入噪声。

```
a = 1.0000001    (存为 1.00000011920929 float32)
b = 1.0000000    (存为 1.00000000000000 float32)

真实差:   0.0000001
算出来:   0.00000011920929

相对误差: 19.2%
```

就一次减法就有 19% 相对误差。ML 里这种情况发生在:
- 算大均值数据的方差: `E[x²] - E[x]²`,E[x] 很大的时候
- 减两个几乎相等的对数概率
- 用太小 epsilon 算有限差分梯度

修法: 重新排列公式,避免减两个大且几乎相等的数。方差用 Welford 算法,或先居中数据。对数概率就在 log 空间算到底。

### Overflow 和 Underflow

Overflow 是结果太大装不下。Underflow 是结果太小(比能表示的最小正数还接近 0)。

```
Float32 边界:
  最大:        3.4028235e+38
  最小正(常规): 1.175e-38
  最小正(非规): 1.401e-45
  Overflow:  任何 > 3.4e38 变 inf
  Underflow: 任何 < 1.4e-45 变 0.0
```

`exp()` 是 ML 里 overflow 的主要来源:

```
exp(88.7)  = 3.40e+38   (在 float32 边缘)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (刚刚在 underflow 之上)
exp(-104)  = 0.0         (下溢成零)
```

`log()` 走相反方向:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (还行)
log(1e-46) = -inf        (输入下溢成 0,然后 log(0) = -inf)
```

ML 里,`exp()` 出现在 softmax、sigmoid、概率计算里。`log()` 出现在交叉熵、对数似然、KL 散度里。`log(exp(x))` 这套组合没招就是雷区。

### Log-Sum-Exp 技巧

直接算 `log(sum(exp(x_i)))` 在数值上很危险。任何 `x_i` 大了,`exp(x_i)` 就会 overflow。所有 `x_i` 都很负,每个 `exp(x_i)` 都 underflow 到 0,`log(0)` 就是 `-inf`。

技巧: 在 exp 之前先减最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为啥这样: 减完 `max(x)` 后,最大的指数是 `exp(0) = 1`。不会 overflow。求和里至少有一项是 1,所以总和至少 1,`log(1) = 0`。不可能 underflow 到 `-inf`。

证明:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (加 c 减 c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (把 exp(c) 提出来)
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设 `c = max(x)`,overflow 就消了。

这个技巧 ML 里到处用:
- Softmax 归一化
- 交叉熵 loss 计算
- 序列模型里的对数概率求和
- 高斯混合
- 变分推断

### 为啥 Softmax 需要减最大 trick

Softmax 把 logits 变概率:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不用 trick 的话,logits [100, 101, 102] 会 overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些在 float32 里 overflow(最大约 3.4e38)?其实:
exp(88.7) 已经到 float32 上限。
exp(100) 在 float32 里是 inf。
```

用 trick,减 max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全一样。算的过程是安全的。这不是优化,是正确性的要求。

### NaN 和 Inf:检测和防止

`nan`(Not a Number) 和 `inf`(无穷)会像病毒一样在计算里传播。一个 `nan` 进梯度更新就让权重变 `nan`,再让之后每个输出都 `nan`。一步之内训练必死。

`inf` 怎么来:
- 大正数的 `exp()`
- 除以零: `1.0 / 0.0`
- float32 累加的 overflow

`nan` 怎么来:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数开 `sqrt()`
- 对负数取 `log()`
- 任何算术里已经存在 `nan`

检测:

```python
import math

math.isnan(x)       # x 是 nan 返回 True
math.isinf(x)       # x 是 +inf 或 -inf 返回 True
math.isfinite(x)    # x 既不是 nan 也不是 inf 返回 True
```

预防策略:
1. 把 `exp()` 的输入 clamp 住: `exp(clamp(x, -80, 80))`
2. 分母加 epsilon: `x / (y + 1e-8)`
3. `log()` 里加 epsilon: `log(x + 1e-8)`
4. 用稳定的实现(log-sum-exp、stable softmax)
5. 梯度裁剪防权重爆炸
6. 调试时每个前向后查一遍 `nan`/`inf`

### 数值梯度检查

解析梯度(从反向传播来的)可能有 bug。数值梯度检查用有限差分来验证它们。

中心差分公式:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h²) 精度,比前向差分 `(f(x+h) - f(x)) / h` 的 O(h) 好得多。

h 选多大: 太大近似不对,太小灾难性 cancellation 毁掉答案。`h = 1e-5` 到 `1e-7` 是常见的。

检查方法: 算解析梯度和数值梯度的相对差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则:
- relative_error < 1e-7: 完美,梯度正确
- relative_error < 1e-5: 可接受,可能正确
- relative_error > 1e-3: 哪里出错了
- relative_error > 1: 梯度完全错

加新层或新 loss 函数时永远要查梯度。PyTorch 提供 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 有专门硬件(Tensor Cores),算 float16 矩阵乘比 float32 快 2-8 倍。混合精度训练就利用这点:

```
1. 维护 float32 主权重
2. 前向用 float16 (快)
3. 用 float32 算 loss (防 overflow)
4. 反向用 float16 (快)
5. 梯度升到 float32
6. 用 float32 主权重做更新
```

纯 float16 训练的问题: 梯度常常非常小(1e-8 或更小)。float16 任何 < 6e-8 的都下溢成零。所有梯度更新都是零,模型停止学习。

修法是 loss scaling:

```
1. loss 乘个大数(比如 1024)
2. 反向算 (loss * 1024) 的梯度
3. 所有梯度大 1024 倍(被推到 float16 范围之上)
4. 更新权重前梯度除以 1024
5. 净效果: 同样的更新,没有下溢
```

动态 loss scaling 自动调缩放因子。从大值(65536)开始。如果梯度溢出成 `inf`,减半。如果连续 N 步不溢出,翻倍。

### bfloat16 vs float16:为啥 bfloat16 训起来更强

```
float16:    [1 符号] [5 指数]  [10 尾数]
bfloat16:   [1 符号] [8 指数]  [7  尾数]
```

float16 精度高(10 bit 尾数 vs 7),但范围小(最大约 65,504)。bfloat16 精度低,但跟 float32 范围一样(最大约 3.4e38)。

训神经网络时:
- 训练高峰时激活和 logits 经常超过 65,504。float16 overflow,bfloat16 没事。
- float16 必须做 loss scaling,bfloat16 通常不用,因为它的范围已经覆盖了梯度量级的全谱。
- bfloat16 就是 float32 截个尾: 把尾数低 16 bit 扔掉。转换简单,指数无损。

float16 适合推理,值有界,精度更重要。bfloat16 适合训练,范围更重要。这就是为啥 TPU 和现代 NVIDIA GPU(A100、H100)原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度经过很多层指数级增长(RNN、深度网络、transformer 常见)。一个大梯度就能一步毁掉所有权重。

两种裁剪:

**按值裁剪:** 独立 clamp 每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但会改变梯度向量的方向。

**按范数裁剪:** 缩放整个梯度向量,让它范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向。这是 `torch.nn.utils.clip_grad_norm_()` 干的事。标准选择。

典型值: transformer 用 `max_norm=1.0`,RL 用 `max_norm=0.5`,简单网络 `max_norm=5.0`。

梯度裁剪不是 hack,是安全机制。没它的话,一个 outlier batch 就能产生足够大的梯度毁掉几周训练。

### 归一化层作为数值稳定器

BatchNorm、LayerNorm、RMSNorm 通常被当作"帮助训练收敛"的正则项。它们也是数值稳定器。

没归一化时,激活值在层与层之间会指数级放大或缩小:

```
第 1 层:  值在 [0, 1]
第 5 层:  值在 [0, 100]
第 10 层: 值在 [0, 10000]
第 50 层: 值在 [0, inf]
```

归一化在每层重新居中和缩放激活值:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`(通常 1e-5)防止所有激活相同时除以零。可学习参数 `gamma` 和 `beta` 让网络恢复任何它需要的尺度。

这让整个网络里值都待在数值安全的范围,前向不 overflow,反向梯度不爆。

### 常见 ML 数值 bug

**Bug: 几个 epoch 后 loss 变 NaN。**
原因: logits 太大,softmax overflow。或者学习率太高权重发散。
修法: 用稳定 softmax(减 max)、降学习率、加梯度裁剪。

**Bug: loss 卡在 log(num_classes)。**
原因: 模型输出接近均匀概率。常常是梯度消失或模型根本没在学。
修法: 检查数据标签对不对、验 loss 函数、查死掉的 ReLU。

**Bug: 验证准确率比预期低 1-3%。**
原因: 混合精度没正确做 loss scaling。梯度下溢默默把小的更新清零。
修法: 开动态 loss scaling,或换 bfloat16。

**Bug: 某些层梯度范数是 0.0。**
原因: ReLU 神经元全死(输入都负)、或 float16 下溢。
修法: 用 LeakyReLU 或 GELU、用梯度缩放、查权重初始化。

**Bug: 模型在一张 GPU 上能用,另一张上结果不一样。**
原因: 非确定性的浮点累加顺序。GPU 并行归约在不同硬件上求和顺序不一样,浮点加法又不是结合的。
修法: 接受小差异(1e-6),或者 `torch.use_deterministic_algorithms(True)` 接受速度损失。

**Bug: loss 计算里 `exp()` 返回 `inf`。**
原因: 原始 logits 直接给了 `exp()`,没减 max。
修法: 用 `torch.nn.functional.log_softmax()`,内部实现了 log-sum-exp。

**Bug: 从 float32 换 float16 训练发散。**
原因: float16 表示不了 6e-8 以下的梯度,或 65,504 以上的激活。
修法: 用混合精度 + loss scaling(AMP),或换 bfloat16。

## Build It

### Step 1: 演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: 实现 naive vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### Step 3: 实现稳定 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### Step 4: 实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: 梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### 模拟混合精度

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保住: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("好", [1.0, 2.0, 3.0])
check_tensor("坏",  [1.0, float('nan'), 3.0])
check_tensor("丑", [1.0, float('inf'), 3.0])
```

完整实现和所有边界情况见 `code/numerical.py`。

## Ship It

本节产出:
- `code/numerical.py`,带稳定 softmax、log-sum-exp、交叉熵、梯度检查、混合精度模拟
- `outputs/prompt-numerical-debugger.md`,排查训练里 NaN/Inf 和数值问题

这些稳定实现在 Phase 3 搭训练循环、Phase 4 实现注意力机制时还会出现。

## Exercises

1. **灾难性 cancellation。** 用 float32 下的 naive 公式 `E[x²] - E[x]²` 算 [1000000.0, 1000001.0, 1000002.0] 的方差。再用 Welford 在线算法算。跟真实方差(0.6667)比误差。
2. **精度猎手。** 找最小的正 float32 值 `x`,使 Python 里 `1.0 + x == 1.0`。这就是机器 epsilon。验证它跟 `numpy.finfo(numpy.float32).eps` 一致。
3. **Log-sum-exp 边界。** 用这些测你的 `logsumexp_stable`: (a) 所有值相等, (b) 一个值远大于其他, (c) 所有值都很负(-1000)。验证它在 naive 版挂的地方给出正确结果。
4. **梯度检查一个神经网络层。** 实现一个线性层 `y = Wx + b` 和它的解析反向传播。用 `numerical_gradient` 给一个 3x2 权重矩阵验正确性。
5. **Loss scaling 实验。** 模拟 float16 训练: 在 [1e-9, 1e-3] 范围造随机梯度,转 float16,量变 0 的比例。再用 loss scaling(乘 1024),转 float16,缩回去,再量变 0 的比例。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "浮点标准" | 国际标准,定义二进制浮点格式、舍入规则、特殊值(inf、nan)。每个现代 CPU/GPU 都实现它。 |
| Machine epsilon | "精度极限" | 某浮点格式下最小的 e,使 1.0 + e != 1.0。float32 约 1.19e-7。 |
| Catastrophic cancellation | "减法的精度损失" | 减两个几乎相等的浮点数,有效位抵消,舍入噪声主导结果。 |
| Overflow | "数太大" | 结果超过最大可表示值变 inf。float32 里 exp(89) 溢出。 |
| Underflow | "数太小" | 结果比最小正数还接近 0,变 0.0。float32 里 exp(-104) 下溢。 |
| Log-sum-exp trick | "先减 max" | 通过把 exp(max(x)) 提出来算 log(sum(exp(x))),防 overflow 和 underflow。用在 softmax、交叉熵、对数概率。 |
| Stable softmax | "不会爆的 softmax" | 指数化前减 max(logits)。数值结果一样,不会 overflow。 |
| Gradient checking | "验证反向传播" | 把解析梯度跟有限差分的数值梯度比,抓实现 bug。 |
| Mixed precision | "前向 float16,反向 float32" | 速度关键的操作用低精度,数值敏感的用高精度。典型加速 2-3 倍。 |
| Loss scaling | "防梯度下溢" | 反向前把 loss 乘个大常数,让梯度待在 float16 范围里,更新权重前再除回来。 |
| bfloat16 | "脑浮点" | Google 的 16 bit 格式,8 bit 指数(跟 float32 一样范围),7 bit 尾数(比 float16 精度低)。训练首选。 |
| Gradient clipping | "卡住梯度范数" | 缩放梯度向量,让范数不超过阈值。防梯度爆炸毁掉权重。 |
| NaN | "不是数" | 来自未定义运算(0/0、inf-inf、sqrt(-1))的特殊浮点值。会在之后所有算术里传播。 |
| Inf | "无穷" | 来自 overflow 或除零的特殊浮点值。能组合产生 NaN(inf - inf、inf * 0)。 |
| Numerical gradient | "暴力求导" | 算 f(x+h) 和 f(x-h) 再除 2h 来近似导数。慢但可靠,用来验证。 |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考,密但全
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 引入 float16 训练 loss scaling 的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 混合精度实战指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为 TPU 选这个格式的原因
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法
