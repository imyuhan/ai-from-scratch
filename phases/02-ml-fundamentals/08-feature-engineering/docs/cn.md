# 特征工程与特征选择

> 好特征顶一千个数据点。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (机器学习统计学、线性代数)、Phase 2 Lessons 1-7
**Time:** ~90 分钟

## 学习目标

- 实现数值变换(标准化、min-max 缩放、对数变换、分箱),说出每种适用的场景
- 为类别特征构造 one-hot、label、target 编码,识别 target 编码的数据泄露风险
- 从零实现 TF-IDF 向量化,解释为什么它比原始词频更适合文本分类
- 用过滤式特征选择(方差阈值、相关性、互信息)降维

## 问题引入

你手头有数据集。挑了个算法。训完。效果一般。换更花哨的算法。还是一般。花了一周调超参。提升一点点。

然后有人把原始数据变成更好的特征,简单的逻辑回归就吊打了你调过的梯度提升集成。

这种事天天都在发生。在经典 ML 里,数据的表示比算法的选择更重要。一个用“面积”和“卧室数”当特征的房价模型,会比一个用“地址字符串”当特征的模型强,不管学习器多复杂。算法只能用它拿到的东西。

特征工程是把原始数据变换成让模型更容易发现规律的表示。特征选择是扔掉那些加噪声不加信号的特征。合起来,这是经典 ML 里**杠杆率最高**的活动。

## 核心概念

### 特征流水线

```mermaid
flowchart LR
    A[原始数据] --> B[处理缺失值]
    B --> C[数值变换]
    B --> D[类别编码]
    B --> E[文本特征]
    C --> F[特征交互]
    D --> F
    E --> F
    F --> G[特征选择]
    G --> H[可直接喂给模型的数据]
```

### 数值特征

原始数字很少能直接喂模型。常用变换:

**缩放:** 把所有特征放到同一区间,这样基于距离的算法(K-Means、KNN、SVM)会一视同仁地对待所有特征。Min-max 缩放映射到 [0, 1]。标准化(z-score)映射到均值 0、方差 1。

**对数变换:** 把右偏分布(收入、人口、词频)压扁。把乘法关系变成加法关系。

**分箱:** 把连续值切成类别。当特征跟目标的关系是非线性但分段时(比如年龄段)特别有用。

**多项式特征:** 造 x²、x³、x1*x2 这种项。让线性模型也能抓非线性关系,代价是特征变多。

### 类别特征

模型要数字。类别要编码。

**One-hot 编码:** 每个类别建一个二值列。"color = red/blue/green" 变成三列:is_red、is_blue、is_green。低基数特征效果好,类别多了就爆炸。

**Label 编码:** 把每个类别映射成整数:red=0、blue=1、green=2。会引入伪序(模型可能觉得 green > blue > red)。只对基于树、能按值切分的模型合适。

**Target 编码:** 用该类别目标变量的均值替换类别。强但危险:数据泄露风险很高。**必须**只在训练数据上算,再用到测试数据上。

### 文本特征

**Count 向量化:** 数每个词在文档里出现几次。"the cat sat on the mat" 变成 {the: 2, cat: 1, sat: 1, on: 1, mat: 1}。

**TF-IDF:** 词频-逆文档频率。按词跨文档的独特程度加权。常用词("the")权低,稀有且有区分力的词权高。

```
TF(词, 文档) = 词在文档中的次数 / 文档总词数
IDF(词) = log(总文档数 / 含该词的文档数)
TF-IDF = TF * IDF
```

### 缺失值

真实数据有洞。处理策略:

- **删除行:** 只在缺失少且随机时
- **均值/中位数填补:** 简单,保持分布形状(中位数对离群点更鲁棒)
- **众数填补:** 用于类别特征
- **指示列:** 填补前先加一个二值列“was_this_missing”。数据缺失这件事本身可能就带信息。
- **前向/后向填充:** 用于时序数据

### 特征交互

规律有时藏在组合里。“身高”和“体重”单独没有“BMI = 体重 / 身高²”预测力强。特征交互会把特征空间扩大,所以要靠领域知识选对交互。

### 特征选择

特征多不一定好。不相关的特征加噪声,拖训练时间,还能引起过拟合。

**过滤式(模型前):**
- 相关性:删掉高度相关的特征(冗余)
- 互信息:衡量知道一个特征能让目标的不确定性降多少
- 方差阈值:删掉几乎不变的特征

**包裹式(基于模型):**
- L1 正则化(Lasso):把无关特征的权重压到正好为 0
- 递归特征消除:训练一次,删最不重要的特征,重复

**为什么选择很重要:** 10 个好特征的模型,通常比“10 个好特征 + 90 个噪声特征”的模型强。噪声特征给模型机会去拟合训练数据里泛化不出来的模式。

```figure
feature-scaling
```

## 从零实现

### Step 1:从零实现数值变换

```python
import math


def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result
```

### Step 2:从零实现类别编码

```python
def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding
```

### Step 3:从零实现文本特征

```python
def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab
```

### Step 4:从零实现缺失值填补

```python
def impute_mean(values):
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]
```

### Step 5:从零实现特征选择

```python
def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]
```

### Step 6:完整流水线和 demo

```python
import random


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== Raw Data Sample ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== Missing Value Handling ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  sqft missing: {sqft_missing}/{len(sqft_raw)}")
    print(f"  age missing: {age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  sqft filled with median: {sqft_fill:.0f}")
    print(f"  age filled with mean: {age_fill:.1f}")

    print("\n=== Numerical Transforms ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  sqft standardized: mean={sum(sqft_scaled)/len(sqft_scaled):.4f}, std={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  age bins: {sorted(set(age_binned))}")

    print("\n=== Categorical Encoding ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  One-hot categories: {ohe_cats}")
    print(f"  Sample encoding: {neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  Label encoding map: {le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  Target encoding: {({k: round(v) for k, v in te_map.items()})}")

    print("\n=== Text Features ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  Vocabulary size: {len(cv_vocab)}")
    print(f"  Doc 0 non-zero features: {sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF vocabulary size: {len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  Doc 0 top TF-IDF words: {top_words}")

    print("\n=== Polynomial Features ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  Input: {[round(v, 4) for v in sample_row]}")
    print(f"  Polynomial: {[round(v, 4) for v in poly]}")
    print(f"  Features: [x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== Feature Selection ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  Total features: {len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  After variance threshold (0.01): {len(surviving_var)} features kept")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  After correlation filter (0.9): {len(surviving_corr)} features kept")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  Mutual information with target:")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}: MI={mi:.4f}")

    print("\n  Correlation with price:")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}: r={corr:.4f}")
```

## 拿来用

用 scikit-learn,这些变换是可组合的流水线:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

从零版本让你看清每个变换里到底在算什么。库版本多出来的是边界处理、稀疏矩阵支持和流水线组合,但数学是一样的。

## 交付物

本课产出:
- `outputs/prompt-feature-engineer.md` —— 一个系统化做特征工程的 prompt

## 练习

1. 给数值变换加鲁棒缩放(用中位数和四分位距代替均值和标准差)。在有极端离群点的数据上跟标准缩放对比。

2. 实现 leave-one-out target encoding:对每行,算排除该行自身目标值后的目标均值。展示这比朴素 target encoding 大幅减少过拟合。

3. 搭一个自动化特征选择流水线,串起方差阈值、相关性过滤、互信息排序。把它跑在房屋数据集上,对比“全特征 vs 选中特征”下简单线性回归的模型性能。

## 关键术语

| 术语 | 大家常说的 | 实际含义 |
|------|-----------|---------|
| Feature engineering | "造新列" | 把原始数据变换成让模型更容易发现规律的表示 |
| Standardization | "让它变正态" | 减均值除以标准差,让特征变成均值 0、方差 1 |
| One-hot encoding | "造哑变量" | 每个类别建一个二值列,每行恰好有一列为 1 |
| Target encoding | "用答案编码" | 用每个类别对应的目标均值替换该类别,带平滑防过拟合 |
| TF-IDF | "高级词频" | 词频乘以逆文档频率:按词跨语料的区分度加权 |
| Imputation | "填空" | 用估计值(均值、中位数、众数或模型预测)替换缺失值 |
| Feature selection | "扔差列" | 删掉加噪声或冗余的特征,只留跟目标有关的 |
| Mutual information | "一件事能告诉你多少另一件" | 衡量观察到变量 X 后,变量 Y 的不确定性减少多少 |
| Data leakage | "不小心作弊" | 训练时用了预测时拿不到的信息,导致结果虚高 |

## 延伸阅读

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/) —— 免费在线书,覆盖特征工程的完整图景
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html) —— 所有标准变换的实战参考
- [Target Encoding Done Right (Micci-Barreca, 2001)](https://dl.acm.org/doi/10.1145/507533.507538) —— 带平滑的 target encoding 原始论文