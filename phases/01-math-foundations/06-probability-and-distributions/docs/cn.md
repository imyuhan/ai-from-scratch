# 概率与分布

> 概率是 AI 表达不确定性的语言。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Learning Objectives

- 从零实现 Bernoulli、Categorical、Poisson、Uniform、Normal 分布的 PMF 和 PDF
- 算期望、方差,用中心极限定理解释为啥高斯分布无处不在
- 实现 softmax 和 log-softmax(带减去最大 logit 的数值稳定技巧)
- 从 logits 算交叉熵 loss,把它跟负对数似然挂上钩

## The Problem

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 5 万个候选里挑下一个词。扩散模型通过从学到的分布里采样生成图像。这些都是概率在干活。

模型的每个预测都是一个概率分布。每个 loss 函数衡量"预测分布离真分布多远"。每步训练都在调参数,让一个分布更接近另一个。不懂概率,你读不懂一篇 ML 论文,调不动一个模型,也不懂为啥训练 loss 会变 NaN。

## The Concept

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率把事件映到 0 到 1 之间的数。

```
掷硬币:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一个骰子:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了整个概率论:
1. 任意事件 A,P(A) >= 0
2. P(S) = 1(事情总会发生点什么)
3. A 和 B 互斥时,P(A 或 B) = P(A) + P(B)

其他一切(Bayes 定理、期望、分布)都从这三条规则推出来。

### 条件概率与独立性

P(A|B) 是在 B 发生的条件下,A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

例子: 一副牌
  P(King | Face card) = P(King 且 Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件独立时,知道一个等于啥也不知道:

```
独立:   P(A|B) = P(A)
等价于: P(A 且 B) = P(A) * P(B)
```

掷硬币独立,不放回抽牌不独立。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数(PMF)。每个结果有一个具体的、能直接读出来的概率。

```
PMF: P(X = k)

公平骰子:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数(PDF)。单点的密度不是概率。概率要在区间上对密度积分才能得到。

```
PDF: f(x)

P(a <= X <= b) = f(x) 从 a 到 b 的积分

f(x) 可以大于 1(密度,不是概率)
f(x) 从 -inf 到 +inf 的积分 = 1
```

这个区别在 ML 里要紧。分类输出是 PMF(离散选择)。VAE 潜空间用 PDF(连续)。

### 常见分布

**Bernoulli:** 一次试验,两个结果。建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**Categorical:** 一次试验,k 个结果。建模多分类(softmax 输出)。

```
P(X = i) = p_i,  p_i 之和 = 1
例子: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** 所有结果等可能。用于随机初始化。

```
离散: P(X = k) = 1/n, k ∈ {1, ..., n}
连续: f(x) = 1/(b-a), x ∈ [a, b]
```

**Normal(高斯):** 钟形曲线。参数是均值 μ 和方差 σ²。

```
f(x) = (1 / sqrt(2πσ²)) * exp(-(x - μ)² / (2σ²))

标准正态: μ = 0, σ = 1
  68% 数据落在 1σ 内
  95% 在 2σ 内
  99.7% 在 3σ 内
```

**Poisson:** 固定区间内稀有事件的计数。建模事件发生率。

```
P(X = k) = (λᵏ * e⁻λ) / k!
均值 = λ,  方差 = λ
```

### 期望与方差

期望是按概率加权的平均结果。

```
离散:   E[X] = x_i * P(X = x_i) 求和
连续:   E[X] = x * f(x) 的积分
```

方差衡量围绕均值的散布程度。

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
标准差 = sqrt(Var(X))
```

ML 里,期望体现为 loss 函数(数据分布上的平均 loss)。方差告诉你模型的稳定性。梯度方差大,训练就吵。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量一起。

联合 PMF 例子(X = 天气, Y = 雨伞):

| | Y=0(不带伞) | Y=1(带伞) | 边缘 P(X) |
|---|---|---|---|
| X=0(晴) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1(雨) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布把另一个变量求和消掉:

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

表里的行/列总和就是边缘。

### 为啥正态分布到处都是

中心极限定理: 很多独立随机变量的和(或均值)收敛到正态分布,跟原始分布无关。

```
掷 1 个骰子:   均匀分布(平的)
2 个骰子的均值: 三角形(尖的)
30 个骰子的均值: 几乎完美的钟形

这个规律对任何起始分布都成立。
```

这就是为啥:
- 测量误差近似正态(很多小的独立来源叠加)
- 神经网络权重初始化用正态分布
- SGD 里的梯度噪声近似正态(很多样本梯度的和)
- 在给定均值和方差的约束下,正态是熵最大的分布

### 对数概率

裸的概率会出数值问题。把很多小概率乘起来,很快就会下溢出成 0。

```
P(句子) = P(word1) * P(word2) * ... * P(word_n)
         = 0.01 * 0.003 * 0.02 * ...
         -> 0.0(大概 30 项后下溢)
```

对数概率能修这个问题。乘法变加法。

```
log P(句子) = log P(word1) + log P(word2) + ... + log P(word_n)
             = -4.6 + -5.8 + -3.9 + ...
             -> 有限数(不会下溢)
```

规则:
- log(a * b) = log(a) + log(b)
- 对数概率永远 <= 0(因为 0 < P <= 1)
- 越负 = 越不可能
- 交叉熵 loss 就是"正确类别的负对数概率"

### Softmax 作为概率分布

神经网络输出原始分数(logits)。Softmax 把它们转成合法的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性质:
  - 所有输出在 (0, 1)
  - 所有输出之和 = 1
  - 保持输入的相对大小关系
  - exp() 放大 logits 之间的差异
```

Softmax 小技巧: 在 exp 之前减掉最大 logit,防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (安全)

结果一样,不会溢出。
```

Log-softmax 把 softmax 和 log 合并到一起,数值更稳。PyTorch 内部算交叉熵 loss 就用它。

### 采样

采样就是从分布里抽随机值。ML 里:
- Dropout 随机采样"把哪些神经元置零"
- 数据增强采样随机变换
- 语言模型从预测分布里采样下一个 token
- 扩散模型采样噪声,逐步去噪

从任意分布里采样要用反变换采样、拒绝采样、重参数化技巧(VAE 用的就是它)这类方法。

## Build It

### Step 1: 概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: 从零写 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: 期望和方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: 从分布里采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax 和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: 中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: 可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现和所有可视化都在 `code/probability.py`。

## Use It

用 NumPy 和 SciPy,上面所有事都是一行:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零搭了这些。现在你知道这些库函数在跑啥了。

## Exercises

1. 给指数分布实现反变换采样。采 10000 个值,跟真实 PDF 比一下直方图。
2. 给两个加权骰子搭一张联合分布表。算边缘分布,看看两个骰子是不是独立的。
3. 一个 5 分类分类器输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`,正确类别是 index 3。算交叉熵 loss。再用 PyTorch 的 `nn.CrossEntropyLoss` 验证。
4. 写一个函数,输入对数概率列表,返回最可能的序列、总对数概率、等价的原始概率。拿 50 个词的句子(每个词概率 0.01)试一下。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "所有可能性" | 一个实验所有可能结果组成的集合 S |
| PMF | "概率函数" | 给每个离散结果精确概率的函数,加起来等于 1 |
| PDF | "概率曲线" | 连续变量的密度函数。在区间上积分才得到概率 |
| Conditional probability | "给定某事下的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和 Bayes 定理的根基 |
| Independence | "互相不影响" | P(A 且 B) = P(A) * P(B)。知道一个事件,啥也不能告诉你另一个 |
| Expected value | "平均值" | 按概率加权的所有结果之和。loss 函数就是一个期望 |
| Variance | "有多散" | 对均值偏差平方的期望。方差大 = 估计噪声大、不稳定 |
| Normal distribution | "钟形曲线" | f(x) = (1/sqrt(2πσ²)) * exp(-(x-μ)²/(2σ²))。因为 CLT 到处都是 |
| Central Limit Theorem | "均值变正态" | 很多独立样本的均值收敛到正态,跟源分布无关 |
| Joint distribution | "两个变量一起" | P(X, Y) 描述 X、Y 每种组合的概率 |
| Marginal distribution | "把另一个变量求和掉" | P(X) = Σ_y P(X, Y)。从联合恢复单个变量的分布 |
| Log probability | "概率的对数" | log P(x)。乘法变加法,长序列里防下溢 |
| Softmax | "把分数变概率" | softmax(z_i) = exp(z_i) / Σ exp(z_j)。把实数 logits 映成合法概率分布 |
| Cross-entropy | "loss 函数" | -Σ(p_true * log(p_predicted))。衡量两个分布的差距,越小越好 |
| Logits | "模型原始输出" | softmax 之前未归一化的分数。名字来自 logistic 函数 |
| Sampling | "抽随机值" | 按概率分布生成值。模型生成输出的方式 |

## Further Reading

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为啥均值变正态的可视化证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖本节所有内容 + 更多的速查
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为啥数值稳定要紧,怎么做到
