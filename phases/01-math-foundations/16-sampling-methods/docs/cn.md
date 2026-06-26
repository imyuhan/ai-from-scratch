# 采样方法

> 采样是 AI 探索"可能性空间"的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- 用均匀随机数从零实现反 CDF、拒绝、重要性采样
- 给语言模型 token 生成搭温度、top-k、top-p(核)采样
- 解释重参数化技巧,以及它为啥让 VAE 能对采样做反向传播
- 跑 Metropolis-Hastings MCMC 从未归一化的目标分布采样

## The Problem

语言模型处理完你的 prompt,产出 50,000 个 logit 的向量。每个词表里的 token 一个。现在它得选一个。咋选?

它总选概率最高的,每次回答都一样。确定性的,无聊的。均匀随机选,输出就是胡话。答案在这俩之间,而这个"之间"靠采样控制。

采样不只用在文本生成。强化学习靠采样轨迹估策略梯度。VAE 通过"从学到的分布采样"+"反向传播穿过随机性"学潜表示。扩散模型靠采样噪声然后逐步去噪生成图。蒙特卡洛估没有闭式解的积分。MCMC 算法探索"枚举不动"的高维后验分布。

每个生成式 AI 系统都是采样系统。采样策略决定输出的质量、多样性、可控性。这节课从零搭每个主流采样方法,从均匀随机数开始,到现代 LLM 和生成模型用的技术收尾。

## The Concept

### 为啥采样重要

采样在 AI 和 ML 里扮演四种根本角色:

**生成。** 语言模型、扩散模型、GAN 都靠采样产出。采样算法直接控制创造性、连贯性、多样性。温度、top-k、核采样是工程师每天拧的旋钮。

**训练。** SGD 采样 mini-batch。Dropout 采样"置零"的神经元。数据增强采样随机变换。重要性采样在强化学习(PPO、TRPO)里重加权样本降梯度方差。

**估计。** ML 里很多量没闭式解。数据分布上的期望 loss、能量模型的配分函数、贝叶斯推断的证据。蒙特卡洛估计把它们都用"对样本取平均"逼近。

**探索。** MCMC 算法探索贝叶斯推断里的后验分布。进化策略采样参数扰动。Thompson sampling 在 bandit 里平衡探索和利用。

核心挑战: 你只能直接从简单分布(均匀、正态)采样。其他所有分布,你都需要一个把"简单样本"变"目标分布样本"的方法。

### 均匀随机采样

所有采样方法都从这里起步。均匀随机数生成器产出 [0, 1) 里的值,每个等长子区间概率相同。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对 0 <= a <= b <= 1

性质:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个离散项里均匀采样,生成 U 然后返回 floor(n * U)。要从连续区间 [a, b] 采样,算 a + (b - a) * U。

关键洞见: 一个均匀随机数正好含产生"任何分布"一个样本所需的随机性。诀窍是找对变换。

### 反 CDF 方法(反变换采样)

累积分布函数(CDF)把值映到概率:

```
F(x) = P(X <= x)

性质:
  F 非减
  F(-inf) = 0
  F(+inf) = 1
  F 把实数映到 [0, 1]
```

反 CDF 把概率映回值。如果 U ~ Uniform(0, 1),那 X = F_inverse(U) 服从目标分布。

```
算法:
  1. 抽 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为啥:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布例子:**

```
PDF: f(x) = λ * exp(-λx),   x >= 0
CDF: F(x) = 1 - exp(-λx)

解 F(x) = u 求 x:
  u = 1 - exp(-λx)
  exp(-λx) = 1 - u
  x = -ln(1 - u) / λ

因为 (1 - U) 和 U 同分布:
  x = -ln(u) / λ
```

能写出 F_inverse 闭式时,这招完美。Normal 分布没有闭式反 CDF,要用别的方法(Box-Muller、或数值近似)。

**离散版:** 对离散分布,建 CDF 当累积和,生成 U,找第一个超过 U 的累积和对应的索引。Lesson 06 的 `sample_categorical` 就是这么干的。

### 拒绝采样

不能反 CDF 但能算目标 PDF(到常数为止)时,拒绝采样能用。

```
目标分布:   p(x)  (能算,可能未归一化)
提议分布:   q(x)  (能采样)
界:        M 满足 p(x) <= M * q(x) 对所有 x

算法:
  1. 抽 x ~ q(x)
  2. 抽 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x)), 接受 x
  4. 否则拒绝,回 step 1

接受率 = 1/M
```

M 越紧,接受率越高。低维(1-3)拒绝采样好使。高维接受率指数级下降,因为大部分提议体积都被拒绝。这就是拒绝采样的维数灾难。

**例子: 从截断正态采样。** 用截断范围上的均匀提议。M 是该范围内正态 PDF 的最大值。

**例子: 从半圆采样。** 在边界矩形上均匀提议。如果点落在半圆内就接受。这就是蒙特卡洛算 pi 的方法: 接受率等于面积比 pi/4。

### 重要性采样

有时你不需要 p(x) 的样本。你需要估"在 p(x) 下的期望",但你手头是另一个分布 q(x) 的样本。

```
目标: 估 E_p[f(x)] = ∫ f(x) * p(x) dx

改写:
  E_p[f(x)] = ∫ f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x)  是重要性权重。

估计量:
  E_p[f(x)] ~ (1/N) * Σ f(x_i) * w(x_i)    其中 x_i ~ q(x)
```

这在强化学习里很关键。PPO(近端策略优化)在老策略 π_old 下收集轨迹,但想优化新策略 π_new。重要性权重是 π_new(a|s) / π_old(a|s)。PPO 裁剪这些权重防新策略离老策略太远。

重要性采样估计量的方差取决于 q 跟 p 多像。q 跟 p 差太多,几个样本就拿到巨大权重并主导估计。自归一化重要性采样用"权重之和"除一下缓解这问题:

```
E_p[f(x)] ~ Σ w_i * f(x_i) / Σ w_i
```

### 蒙特卡洛估计

蒙特卡洛估计通过对随机样本取平均来逼近积分。大数定律保证收敛。

```
目标: 估 I = ∫ g(x) dx,定义域 D

方法:
  1. 从 D 均匀抽 x_1, ..., x_N
  2. I ~ (D 体积 / N) * Σ g(x_i)

误差: O(1 / sqrt(N))   跟维度无关
```

误差率跟维度无关。这就是蒙特卡洛方法在高维占主导的原因 —— 在那里基于网格的积分不可能。

**估 pi:**

```
从 [-1, 1] × [-1, 1] 抽 (x, y)
数有几个落单位圆里: x² + y² <= 1
pi ~ 4 * (落在里头的) / (总数)
```

**估期望:**

```
E[f(X)] ~ (1/N) * Σ f(x_i)    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛(MCMC):Metropolis-Hastings

MCMC 构造一条马尔可夫链,让它的平稳分布是目标分布 p(x)。走够多步之后,链上的样本(近似)就是 p(x) 的样本。

```
目标:  p(x)  (已知到归一化常数)
提议:  q(x'|x)  (给定当前状态怎么提议下一个)

Metropolis-Hastings 算法:
  1. 从某 x_0 起步
  2. 对 t = 1, 2, ..., T:
     a. 提议 x' ~ q(x'|x_t)
     b. 算接受比:
        α = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, α) 接受:
        - 如果 u < α (u ~ Uniform(0,1)): x_{t+1} = x'
        - 否则: x_{t+1} = x_t
  3. 扔前 B 个样本(burn-in)
  4. 返回剩下的样本
```

对称提议(q(x'|x) = q(x|x'))时,比化简为 p(x')/p(x)。这是最初的 Metropolis 算法。

**为啥。** 接受规则保证细致平衡: 处于 x 移到 x' 的概率等于处于 x' 移到 x 的概率。细致平衡蕴含 p(x) 是链的平稳分布。

**实际考虑:**
- Burn-in: 扔链到达平衡前的早期样本
- Thinning: 每 k 个样本留一个降自相关
- 提议尺度: 太小链挪得慢(高接受、慢探索);太大多数提议被拒(低接受、原地卡死)
- 高维高斯提议的最优接受率约 0.234

### Gibbs 采样

Gibbs 采样是 MCMC 的一种特殊情况,针对多维分布。它不"一次性在所有维度提议",而是一次从一个变量,从条件分布更新。

```
目标:  p(x_1, x_2, ..., x_d)

算法:
  对每次迭代 t:
    抽 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    抽 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    抽 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样需要你能从每个条件分布 p(x_i | x_{-i}) 采样。很多模型这都是直接的:
- 贝叶斯网络: 条件分布从图结构来
- 高斯混合: 条件是高斯
- Ising 模型: 每个自旋的条件只依赖邻居

接受率永远是 1(每个提议都被接受),因为从精确条件采样自动满足细致平衡。

**局限。** 变量高度相关时,Gibbs 采样混合慢,因为"一次变一个"不能在对角线方向走大跨步。

### 温度采样(LLM 里用)

语言模型对词表每个 token 输出 logits z_1, ..., z_V。Softmax 把它们变概率。温度在 softmax 前对 logits 重新缩放:

```
p_i = exp(z_i / T) / Σ exp(z_j / T)

T = 1.0: 标准 softmax(原分布)
T -> 0:  argmax(确定性,总选最高 logit)
T -> ∞: 均匀(所有 token 等可能)
T < 1.0: 让分布更尖(更自信,更不多样)
T > 1.0: 让分布更平(更不自信,更多样)
```

**为啥。** 除以 T < 1 放大 logits 之间的差。如果 z_1 = 2, z_2 = 1,除以 T = 0.5 得 z_1/T = 4, z_2/T = 2,差变大了。softmax 之后,最高 logit 的 token 占更大份额。

**实际中:**
- T = 0.0: 贪心解码,事实性问答最佳
- T = 0.3-0.7: 略创造,代码生成不错
- T = 0.7-1.0: 平衡,通用对话
- T = 1.0-1.5: 创意写作、头脑风暴
- T > 1.5: 越来越随机,很少有用

温度不改变哪些 token 是可能的。它改变分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样把候选集限制到概率最高的 k 个 token,然后重新归一并从限制集里采样。

```
算法:
  1. 对所有 V 个 token 算 softmax 概率
  2. 按概率降序排 token
  3. 只留 top k 个
  4. 重新归一: p_i' = p_i / Σ p_j (j 在 top-k)
  5. 从重新归一的分布采样

k = 1:  贪心解码
k = V:  不过滤(标准采样)
k = 40: 典型设置,移走长尾里不太可能的 token
```

Top-k 防止模型选中"极不可能的 token"(打错字、胡话),这些在词表分布的长尾里。问题: k 是固定的,不随上下文。模型自信(一个 token 95% 概率)时,k = 40 还允许 39 个备选。模型不确定(概率撒在 1000 个 token 上)时,k = 40 切掉了合理选项。

### Top-p(核)采样

Top-p 采样动态调整候选集大小。它不留固定数量的 token,留"累计概率超过 p 的最小集合"。

```
算法:
  1. 对所有 V 个 token 算 softmax 概率
  2. 按概率降序排 token
  3. 找最小 k 使 top-k 概率之和 >= p
  4. 只留那 k 个
  5. 重新归一 + 采样

p = 0.9:  留覆盖 90% 概率质量的 token
p = 1.0:  不过滤
p = 0.1:  极严,几乎贪心
```

模型自信时,核采样留很少 token(可能 2-3 个)。模型不确定时,留很多(可能 200)。这种自适应行为让核采样通常比 top-k 出更好的文本。

**常见组合:**
- 温度 0.7 + top-p 0.9: 通用好设置
- 温度 0.0(贪心): 确定性任务最佳
- 温度 1.0 + top-k 50: Fan et al. (2018) 原论文设置

Top-k 和 top-p 可以组合。先用 top-k,再在剩下的上用 top-p。

### 重参数化技巧(VAE 里用)

变分自编码器(VAE)把输入编码到潜空间的一个分布、从该分布采样、把样本解码回来,从而学习。问题: 你不能对采样操作反向传播。

```
标准采样(不可微):
  z ~ N(μ, σ²)

  随机性挡住梯度流。
  d/d_μ [sample from N(μ, σ²)] = ???
```

重参数化技巧把随机性从参数里分开:

```
重参数化采样:
  ε ~ N(0, 1)           (固定随机噪声,没参数)
  z = μ + σ * ε        (参数的可微函数)

  现在 z 是 μ、σ 的可微函数。
  d(z)/d(μ) = 1
  d(z)/d(σ) = ε

  梯度流过 μ 和 σ。
```

这能成是因为 N(μ, σ²) 跟 μ + σ * N(0, 1) 同分布。关键洞见: 把随机性挪到"无参数源"(ε),然后把样本表达成参数的可微变换。

**VAE 训练循环里:**
1. 编码器对每个输入输出 μ 和 log(σ²)
2. 采 ε ~ N(0, 1)
3. 算 z = μ + σ * ε
4. 解码 z 重建输入
5. 沿 4、3、2、1 反向传播(因为第 3 步可微,能跑通)

没重参数化技巧,VAE 用标准反向传播训不了。这一个洞见让 VAE 变得实用。

### Gumbel-Softmax(可微类别采样)

重参数化技巧对连续分布(高斯)管用。对离散类别分布,你需要别的方法。Gumbel-Softmax 给类别采样的可微近似。

**Gumbel-Max 技巧(不可微):**

```
从类别分布抽(对数概率 log p_1, ..., log p_k):
  1. 对每个类别抽 g_i ~ Gumbel(0, 1)
     (g = -log(-log(u)),其中 u ~ Uniform(0, 1))
  2. 返回 argmax(log(p_i) + g_i)

这产出精确的类别样本。
```

**Gumbel-Softmax(可微近似):**

```
把硬 argmax 换成软 softmax:
  y_i = exp((log(p_i) + g_i) / τ) / Σ exp((log(p_j) + g_j) / τ)

τ (温度) 控制近似:
  τ -> 0:  趋近 one-hot(硬类别)
  τ -> ∞: 趋近均匀(1/k, 1/k, ..., 1/k)
  τ = 1.0: 软近似
```

Gumbel-Softmax 产出离散样本的连续松弛。输出是概率向量(软 one-hot)而不是硬 one-hot。梯度流过 softmax。训练前向时,可以用 "straight-through" 估计器: 前向用硬 argmax,反向用软 Gumbel-Softmax 梯度。

**应用:**
- VAE 里的离散潜变量
- 神经架构搜索(选离散操作)
- 硬注意力机制
- 离散动作的强化学习

### 分层采样

标准蒙特卡洛采样会偶然在样本空间留空隙。分层采样强制均匀覆盖: 把空间分成层,从每层采样。

```
标准蒙特卡洛:
  从 [0, 1] 均匀抽 N 个点
  一些区域可能挤,一些有缝

分层采样:
  把 [0, 1] 分成 N 个等层: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每层里均匀采一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样总是比标准蒙特卡洛方差更低或相等:

```
Var(分层) <= Var(标准蒙特卡洛)

f(x) 越平滑改进越大。
分段常数函数,分层采样就是精确的。
```

**应用:**
- 数值积分(准蒙特卡洛)
- 训练数据划分(保证每折类别平衡)
- 分层 + 重要性采样(两种技术结合)
- NeRF(神经辐射场)沿相机射线用分层采样

### 跟扩散模型的联系

扩散模型通过采样过程生成图。正向过程 T 步往图上加高斯噪声直到变纯噪声。逆向过程学去噪,一步步恢复原图。

```
正向过程(已知):
  x_t = sqrt(α_t) * x_{t-1} + sqrt(1 - α_t) * ε
  其中 ε ~ N(0, I)

  T 步后: x_T ~ N(0, I)  (纯噪声)

逆向过程(学到):
  x_{t-1} = (1/sqrt(α_t)) * (x_t - (1 - α_t)/sqrt(1 - ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
  其中 z ~ N(0, I)

  每个去噪步都是一步采样。
```

跟这节课方法的关系:
- 每步去噪用重参数化技巧(采噪声,做确定变换)
- 噪声 schedule {α_t} 控制一种"温度退火"
- 训练用蒙特卡洛估近似 ELBO(证据下界)
- 扩散模型里的 Ancestral 采样是条马尔可夫链(每步只依赖当前状态)

整个图生成过程就是迭代采样: 从噪声开始,每步在"学到的去噪模型"条件下采一个稍没那么噪声的版本。

## Build It

### Step 1: 均匀和反 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数样本,验均值约 1/λ。

### Step 2: 拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态采。画直方图验形状。

### Step 3: 重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀提议估正态下的 E[X²]。跟已知答案(μ² + σ²)对。

### Step 4: 蒙特卡洛估 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布(两个高斯的混合)采。画链的轨迹。

### Step 6: Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: 温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度怎么改一组 token logits 的输出分布。

### Step 8: Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: 重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

展示梯度流过重参数化样本,但不流过直接采样。

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示温度降下来,输出趋近 one-hot。

完整实现和所有可视化在 `code/sampling.py`。

## Use It

用 NumPy 和 SciPy,生产版本:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数均值: {exponential_samples.mean():.4f} (期望 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"1.96 处的 CDF: {normal.cdf(1.96):.4f}")
print(f"0.975 处的反 CDF: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采到的 token 索引: {token}")
```

大规模 MCMC 用专门的库:
- PyMC: 完整贝叶斯建模 + NUTS(自适应 HMC)
- emcee: 集成 MCMC 采样器
- NumPyro/JAX: GPU 加速 MCMC

你从零搭了这些。现在你知道这些库调用在跑啥了。

## Exercises

1. 给 Cauchy 分布实现反 CDF 采样。CDF 是 F(x) = 0.5 + arctan(x)/π。生成 10,000 个样本,把直方图画到真 PDF 上。注意重尾(远离中心的极值)。
2. 用拒绝采样,从 Uniform(0, 1) 提议,从 Beta(2, 5) 分布采。把接受的样本画到真 Beta PDF 上。理论接受率多少?
3. 用 1,000、10,000、100,000 样本蒙特卡洛估 sin(x) 从 0 到 π 的积分。每个层级的误差比一比。验证误差按 O(1/√N) 缩。
4. 实现 Metropolis-Hastings 从 2D 分布 p(x, y) ∝ exp(-(x²y² + x² + y² - 8x - 8y) / 2) 采。画样本和链的轨迹。试不同提议标准差。
5. 搭完整文本生成 demo: 给 10 个词的词表带 logits,用 (a) 贪心, (b) 温度=0.7, (c) top-k=3, (d) top-p=0.9 生成 20 token 的序列。5 跑间比输出多样性。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "抽随机值" | 按概率分布生成值。所有生成式 AI 的机制 |
| Uniform distribution | "全都等可能" | [a, b] 上每个值概率密度 1/(b-a) 都相等。所有采样的起点 |
| Inverse CDF | "概率变换" | F_inverse(U) 把均匀样本变任何有已知 CDF 分布的样本。精确高效 |
| Rejection sampling | "提议然后接受/拒绝" | 从简单提议采,以正比于 target/proposal 的概率接受。精确但费样本 |
| Importance sampling | "重加权样本" | 用 q(x) 的样本,通过 p(x)/q(x) 加权估 p(x) 下的期望。RL 里 PPO 的核心 |
| Monte Carlo | "随机样本取平均" | 积分用样本平均近似。误差 O(1/√N),跟维度无关 |
| MCMC | "收敛的随机游走" | 构造马尔可夫链,平稳分布是目标。Metropolis-Hastings 是奠基算法 |
| Metropolis-Hastings | "接受上山,偶尔下山" | 提议移动,按密度比接受。细致平衡保证收敛到目标分布 |
| Gibbs sampling | "一次变一个" | 从条件分布更新每个变量,其他固定。接受率 100% |
| Temperature | "自信度旋钮" | softmax 前除以 T。T<1 锐化(更自信),T>1 平坦(更多样) |
| Top-k sampling | "留 k 个最好的" | 概率最高 k 个 token 之外全置零,重新归一,采样。固定候选集 |
| Nucleus sampling (top-p) | "留那些可能的" | 留累计概率超 p 的最小 token 集。自适应候选集大小 |
| Reparameterization trick | "把随机性挪外面" | 写 z = μ + σε, ε ~ N(0,1)。让采样可微。VAE 训练必须 |
| Gumbel-Softmax | "软类别采样" | 用 Gumbel 噪声 + 温度 softmax 的类别采样可微近似 |
| Stratified sampling | "强制覆盖" | 把样本空间分到层,从每层采。比天真蒙特卡洛方差永远更低 |
| Burn-in | "热身期" | 早期 MCMC 样本,链到达平稳分布前扔掉 |
| Detailed balance | "可逆条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 是马尔可夫链平稳分布的充分条件 |
| Diffusion sampling | "迭代去噪" | 从噪声开始,应用学到的去噪步生成数据。每步都是条件采样操作 |

## Further Reading

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核(top-p)采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE 论文,引入重参数化技巧
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 把采样跟图生成挂上钩
