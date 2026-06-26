# 朴素贝叶斯

> “朴素”的假设是错的,但它就是管用。这就是它的美。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (分类、Bayes 定理)
**Time:** ~75 分钟

## 学习目标

- 从零实现带 Laplace 平滑的多项式朴素贝叶斯,用于文本分类
- 解释朴素独立性假设为什么数学上错误,但实际产出正确的类排名
- 对比多项式、伯努利、高斯三种朴素贝叶斯变体,为给定特征类型选最合适的
- 在高维稀疏数据上对比朴素贝叶斯和逻辑回归,解释背后的偏差-方差权衡

## 问题引入

你要做文本分类。把邮件分垃圾邮件/正常邮件,客户评价分正/负,工单分类。你有上千个特征(每个词一个),训练数据又少。

大多数分类器在这里栽跟头。逻辑回归需要足够多的样本来可靠地估上千个权重。决策树一次只切一个词,严重过拟合。KNN 在 10,000 维里没意义,因为每个点跟每个点的距离都一样远。

朴素贝叶斯能干这个。它做一个数学上错的假设(每个特征在给定类下相互独立),结果在文本分类上,特别是小训练集下,反而吊打“更聪明”的模型。一次遍历数据就能训。扩展到百万级特征。会产出概率估计(虽然因为独立性假设,校准往往不太好)。

理解为什么错的假设能产出好预测,会教你一件关于机器学习的基本道理:最好的模型不是最正确的模型,是那个对你的数据偏差-方差权衡最优的模型。

## 核心概念

### Bayes 定理(快速回顾)

Bayes 定理翻转条件概率:

```
P(类 | 特征) = P(特征 | 类) * P(类) / P(特征)
```

我们想要 `P(类 | 特征)`——文档属于某个类的概率,给定它含的词。我们能从这些算出来:
- `P(特征 | 类)`——在这个类的文档里看到这些词的似然
- `P(类)`——类的先验概率(垃圾邮件一般有多常见?)
- `P(特征)`——证据,对所有类都一样,所以比较时能忽略

`P(类 | 特征)` 最大的类胜出。

### 朴素独立性假设

精确算 `P(特征 | 类)` 需要估计所有特征的联合概率。词表 10,000 词时,你需要估计 2^10,000 种可能组合的分布。不可能。

朴素假设:每个特征在给定类下条件独立。

```
P(w1, w2, ..., wn | 类) = P(w1 | 类) * P(w2 | 类) * ... * P(wn | 类)
```

不是估一个不可能的联合分布,而是估 n 个简单的单特征分布。每个只要一个计数。

这个假设明显是错的。文档里“machine”和“learning”两词根本不独立。但分类器不需要正确的概率估计,它需要正确的排名——哪个类的概率最高。独立性假设引入系统误差,但这些误差对所有类影响类似,所以排名保持正确。

### 为什么它还是管用

三个原因:

1. **排名比校准重要**。分类只要排在最前面的类对就行。就算 P(垃圾) = 0.99999 而真实概率是 0.7,分类器还是选垃圾邮件。我们不需要正确概率,需要正确赢家。

2. **高偏差低方差**。独立性假设是个强先验。它把模型约束得很死,防止过拟合。数据有限时,一个稍错但稳定的模型,比理论上对但剧烈波动的模型强。这就是偏差-方差权衡的实际案例。

3. **特征冗余会抵消**。相关特征提供冗余证据。分类器把这份证据算了两次,但对正确的类也是算了两次。如果“machine”和“learning”总是一起出现,两个都给“tech”类提供证据。NB 算了它们两次,但对正确的类算了两次。

第四个实际原因:朴素贝叶斯极快。训练就是遍历数据数一次频次。预测是个矩阵乘法。你能在几秒内训百万文档。这个速度让你能更快迭代、试更多特征集、跑比慢模型更多的实验。

### 数学一步步来

我们走一个具体例子。假设两类:垃圾邮件和非垃圾邮件。词表三个词:“free”、“money”、“meeting”。

训练数据:
- 垃圾邮件提到“free”80 次、“money”60 次、“meeting”10 次(总共 150 词)
- 非垃圾邮件提到“free”5 次、“money”10 次、“meeting”100 次(总共 115 词)
- 40% 邮件是垃圾邮件,60% 是非垃圾邮件

带 Laplace 平滑(alpha=1):

```
P(free | 垃圾)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | 垃圾)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | 垃圾) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | 非垃圾)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | 非垃圾)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | 非垃圾) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件含:“free”(2 次)、“money”(1 次)、“meeting”(0 次)。

```
log P(垃圾 | 邮件) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                   = -0.916 + 2*(-0.637) + (-0.919) + 0
                   = -3.109

log P(非垃圾 | 邮件) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                     = -0.511 + 2*(-2.976) + (-2.375) + 0
                     = -8.838
```

垃圾邮件以大优势胜出。“free”出现两次是垃圾邮件的强证据。注意“meeting”不出现对两个对数求和贡献为 0(0 * log(P))——多项式 NB 里,缺席词没影响。是伯努利 NB 显式建模词的缺席。

### 三种变体

朴素贝叶斯有三种口味。每种对 `P(特征 | 类)` 的建模不同。

#### 多项式朴素贝叶斯

把每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | 类) = (该类中 word_i 的计数 + alpha) / (该类总词数 + alpha * 词表大小)
```

`alpha` 是 Laplace 平滑(下面解释)。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯

把每个特征建模为正态分布。最适合连续特征。

```
P(x_i | 类) = (1 / sqrt(2 * pi * 方差)) * exp(-(x_i - 均值)² / (2 * 方差))
```

每个类在每个特征上拿自己的均值和方差。特征在每个类里真的呈钟形曲线时这招很灵。

#### 伯努利朴素贝叶斯

把每个特征建模为二值(出现或不出现)。最适合短文本或二值特征向量。

```
P(word_i | 类) = (该类中含 word_i 的文档数 + alpha) / (该类总文档数 + 2 * alpha)
```

跟多项式不同,伯努利显式惩罚词的缺席。如果“free”通常在垃圾邮件里出现,但这封邮件里没出现,伯努利把这算成反对垃圾邮件的证据。

### 每种变体什么时候用

| 变体 | 特征类型 | 最适合 | 例子 |
|------|---------|--------|------|
| 多项式 | 计数或频率 | 文本分类、词袋 | 邮件垃圾、主题分类 |
| 高斯 | 连续值 | 特征近似正态的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利 | 二值 (0/1) | 短文本、二值特征向量 | 短信垃圾、出现/缺席特征 |

### Laplace 平滑

当测试数据里出现了某个词,但在某个类的训练数据里从没出现过,会怎样?

不平滑:`P(词 | 类) = 0/N = 0`。整个乘积里乘一个零就让 `P(类 | 特征) = 0`,不管其他证据多强。一个没见过的词毁掉整个预测,无论其他证据多支持这个类。

Laplace 平滑给每个特征计数加一个小数 `alpha`(通常 1):

```
P(word_i | 类) = (count(word_i, 类) + alpha) / (该类总词数 + alpha * 词表大小)
```

alpha=1 时,每个词都至少有一点小概率。测试邮件里出现“discombobulate”不再把垃圾邮件概率干掉。这个平滑有个 Bayesian 解释:等价于在词分布上放一个均匀 Dirichlet 先验。

alpha 越大平滑越强(分布更均匀)。alpha 越小模型越信任数据。alpha 是个要调的超参。

alpha 的效果:

| Alpha | 效果 | 何时用 |
|-------|------|--------|
| 0.001 | 几乎不平滑,信任数据 | 训练集很大,不会出现未见特征 |
| 0.1 | 轻平滑 | 训练集大 |
| 1.0 | 标准 Laplace 平滑 | 默认起点 |
| 10.0 | 重平滑,分布被压平 | 训练集很小,很多未见特征 |

### 对数空间计算

乘几百个概率(每个都小于 1)会引起浮点下溢。乘积在浮点里变成零,即使真实值是个很小的正数。

解决:在对数空间里算。不乘概率,而是加对数:

```
log P(类 | x1, x2, ..., xn) = log P(类) + sum_i log P(xi | 类)
```

这把预测变成点积:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测那么快的原因——跟单层线性模型是同一回事。

### 朴素贝叶斯 vs 逻辑回归

两者都是文本线性分类器。区别在建模的东西。

| 维度 | 朴素贝叶斯 | 逻辑回归 |
|------|----------|---------|
| 类型 | 生成式(建模 P(X\|Y)) | 判别式(建模 P(Y\|X)) |
| 训练 | 数频次 | 优化损失函数 |
| 小数据 | 更好(强先验有帮助) | 较差(样本不够估权重) |
| 大数据 | 较差(错的假设开始碍事) | 更好(灵活边界) |
| 特征 | 假设独立 | 能处理相关性 |
| 速度 | 单遍,极快 | 迭代优化 |
| 校准 | 概率差 | 概率好 |

经验法则:从朴素贝叶斯开始。如果你有足够数据而且 NB 平台期,换逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[建词表]
    C --> D[数词频]
    D --> E[应用平滑]
    E --> F[算对数概率]
    F --> G[预测: argmax P(类|词)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实际中我们用对数空间避免下溢。不乘很多小概率,加它们的对数:

```
log P(类 | 特征) = log P(类) + sum_i log P(特征_i | 类)
```

```figure
naive-bayes
```

## 从零实现

`code/naive_bayes.py` 从零实现 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现:

1. **fit(X, y)**:对每个类,数每个特征的频率。加 Laplace 平滑。算对数概率。存类先验(类频率的对数)。

2. **predict_log_proba(X)**:对每个样本,对所有类算 log P(类) + sum log P(特征_i | 类)。这是矩阵乘法:X @ log_probs.T + log_priors。

3. **predict(X)**:返回对数概率最高的类。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞见:fit 之后,预测就是矩阵乘法加偏置。这就是朴素贝叶斯那么快的原因。

### GaussianNB

对连续特征,我们对每类每特征估均值和方差:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测用每特征的高斯 PDF,在所有特征上乘起来(对数空间里加)。

### Demo:文本分类

代码生成模拟两类的合成词袋数据(科技文章 vs 体育文章)。每个类有不同的词频分布。MultinomialNB 用词频分类。

合成数据这样工作:造 200 个“词”(特征列)。词 0-39 在科技文章里高频,在体育文章里低频。词 80-119 在体育文章里高频,在科技文章里低频。词 40-79 在两个里都是中频。这创造了一个现实的场景,有些词是强类指示器,有些是噪声。

### Demo:连续特征

代码生成鸢尾花风格的数据(3 类、4 特征、高斯簇)。GaussianNB 用每类的均值和方差分类。每个类有不同的中心(均值向量)和不同的散布(方差),模拟真实世界中测量在不同类别间系统性差异。

代码还演示:
- **平滑对比**:用不同 alpha 训 MultinomialNB,展示平滑强度对准确率的影响
- **训练规模实验**:NB 准确率随训练数据从 20 涨到 1600 怎么变。NB 极少样本就能拿到不错准确率——这是它主要优势
- **混淆矩阵**:每类的精确率、召回率、F1,看 NB 在哪犯错

### 预测速度

朴素贝叶斯预测是矩阵乘法。对 n 样本、d 特征、k 类:
- MultinomialNB:一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB:n * k 次高斯 PDF 评估,每次 d 特征 = O(n * d * k)

两个在每个维度上都是线性的。跟 KNN(要算到所有训练点的距离)或带 RBF 核的 SVM(要算对所有支持向量的核)比比,NB 在预测时快了几个数量级。

## 拿来用

用 sklearn,两个变体都是一行调用:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

sklearn 文本分类:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 里的代码在同一数据上对比从零实现跟 sklearn,验证正确性。

### NB + TF-IDF

原始词频给每个词每次出现同等权重。但像“the”、“is”这种常用词在每类都高频出现——它们不带信息。TF-IDF(词频-逆文档频率)压低常用词的权重,抬高稀有且有区分力的词。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负,所以能配 MultinomialNB。TF-IDF + MultinomialNB 是文本分类最强的基线之一。在不到 10,000 训练样本的数据集上经常吊打更复杂的模型。

### BernoulliNB 给短文本

短文本(推文、短信、聊天消息)用 BernoulliNB 可能比 MultinomialNB 强。短文本词数少,MultinomialNB 依赖的频率信息噪声大。BernoulliNB 只关心出现不出现,在短文本上更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 里的 `binary=True` 把所有计数转成 0/1。不开它的话 BernoulliNB 也能跑,但它看到的不是为它设计的计数。

### 校准 NB 的概率

NB 的概率校准差。NB 说 P(垃圾) = 0.95 时,真实概率可能是 0.7。如果你需要可靠的概率估计(比如设阈值或跟其他模型合用),用 sklearn 的 CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它用交叉验证在 NB 的原始分数上套一层逻辑回归。得到的概率更接近真实类频率。

### 常见坑

1. **负特征值**。MultinomialNB 要求非负特征。如果你有负值(比如某些设置下的 TF-IDF 或标准化后的特征),改用 GaussianNB,或把特征平移到非负。

2. **零方差特征**。GaussianNB 要除以方差。如果某特征在某类下方差为零(所有值相同),概率计算就崩。代码给所有方差加一个小平滑项(1e-9)防这个。

3. **类别不平衡**。如果 99% 邮件非垃圾,先验 P(非垃圾) = 0.99 就把似然证据压没了。你能手动设类先验,或在 sklearn 里用 class_prior 参数。

4. **特征缩放**。MultinomialNB 不需要缩放(它处理计数)。GaussianNB 也不需要(它估每特征统计)。这相对逻辑回归和 SVM 是个优势,后两个对特征量纲敏感。

## 交付物

本课产出:
- `outputs/skill-naive-bayes-chooser.md` —— 一个帮你选 NB 变体的决策 skill
- `code/naive_bayes.py` —— 从零的 MultinomialNB 和 GaussianNB,带 sklearn 对比

### 朴素贝叶斯失败的时候

NB 失败于独立性假设导致排名错(不只是概率错)的时候。这种情况发生在:

1. **强特征交互**。如果类取决于两个特征的组合但不取决于任一单独(XOR 模式),NB 会彻底错过。每个特征单独没证据,NB 不能非线性组合它们。

2. **高度相关的特征给相反证据**。如果特征 A 说“垃圾”、特征 B 说“非垃圾”,但 A 和 B 完全相关(现实中总是一致),NB 会看到实际上不存在的冲突证据。

3. **训练集特别大**。数据足够时,逻辑回归这种判别式模型学到了真实决策边界,吊打 NB。独立性假设在小数据时是帮忙,在大数据时就成限制了。

实际中文本分类很少遇这些坑。文本特征多、单个弱、独立性假设的误差倾向于抵消。表格数据特征少且强相关时,先考虑逻辑回归或树模型。

## 练习

1. **平滑实验**。在文本数据上用 alpha = 0.01、0.1、1.0、10.0、100 训 MultinomialNB。画准确率对 alpha 的曲线。性能峰值在哪?为什么 alpha 特别高会掉?

2. **特征独立性测试**。拿一个真实文本数据集。选两个明显相关的词(“machine”和“learning”)。算 P(词1 | 类) * P(词2 | 类),跟 P(词1 AND 词2 | 类) 比。独立性假设差多少?影响分类准确率吗?

3. **伯努利实现**。给代码加一个 BernoulliNB 类。把词袋转成二值(出现/缺席),在文本数据上跟 MultinomialNB 比准确率。伯努利什么时候赢?

4. **NB vs 逻辑回归**。在文本数据上都训。从 100 训练样本开始涨到 10,000。画准确率对训练集大小,两个都画。逻辑回归在哪个点超过朴素贝叶斯?

5. **垃圾邮件过滤器**。搭一个完整垃圾邮件分类器:对原始邮件文本分词、建词表、造词袋特征、训 MultinomialNB、用精确率和召回率评估(不只是准确率——为什么?)。

## 关键术语

| 术语 | 大家常说的 | 实际含义 |
|------|-----------|---------|
| Naive Bayes | "简单概率分类器" | 用 Bayes 定理加特征在给定类下条件独立的假设做分类 |
| Conditional independence | "特征互不影响" | P(A, B \| C) = P(A \| C) * P(B \| C)——知道 C 后 B 对 A 没说新信息 |
| Laplace smoothing | "加一平滑" | 给每个特征加一个小计数,防零概率主宰预测 |
| Prior | "看数据前你信的" | P(类)——观察任何特征前每个类的概率 |
| Likelihood | "数据多符合" | P(特征 \| 类)——类已知时观察到这些特征的概率 |
| Posterior | "看数据后你信的" | P(类 \| 特征)——观察到特征后类的更新概率 |
| Generative model | "建模数据怎么生成" | 学 P(X \| Y) 和 P(Y),再用 Bayes 定理得 P(Y \| X) |
| Discriminative model | "建模决策边界" | 直接学 P(Y \| X),不建模 X 怎么生成 |
| Log probability | "防下溢" | 用 log P 而不是 P,防止很多小数相乘在浮点里变零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 三种变体的数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 多项式 vs 伯努利在文本上的经典对比
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— NB 在文本上的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明 NB 比 LR 收敛更快,数据更少时