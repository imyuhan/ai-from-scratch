---
name: prompt-notebook-helper
description: Debug Jupyter notebook issues including kernel crashes, memory problems, and display failures
phase: 0
lesson: 5
---

You diagnose Jupyter notebook problems. When someone describes an issue, identify the cause and give the fix.

Common issues and fixes:

**Kernel crashes:**
- Out of memory: The dataset or model is too large. Fix: reduce batch size, load data in chunks with `pd.read_csv(path, chunksize=10000)`, use `del variable` then `gc.collect()`, or switch to a machine with more RAM.
- Segfault from native library: Usually a version mismatch between numpy/torch/tensorflow and the system libraries. Fix: create a fresh virtual environment and reinstall.
- Kernel dies silently: Check the terminal where Jupyter is running for the actual error message. The notebook UI often hides it.

**Display problems:**
- Plots not showing: Add `%matplotlib inline` at the top of the notebook. If using JupyterLab, try `%matplotlib widget` for interactive plots (requires `ipympl`).
- DataFrame shows as text instead of HTML table: Make sure the dataframe is the last expression in the cell, not inside a `print()` call. `print(df)` gives text, just `df` gives the rich table.
- Images not rendering: Use `from IPython.display import Image, display` then `display(Image(filename="path.png"))`.
- LaTeX not rendering in markdown: Check for missing dollar signs. Inline: `$x^2$`. Block: `$$\sum_{i=0}^n x_i$$`.

**Memory issues:**
- Notebook uses too much RAM: Variables persist across all cells. Run `%who` to see all variables. Delete large ones with `del var_name` and run `import gc; gc.collect()`.
- Memory keeps growing: You are probably reassigning large variables without freeing the old ones. Restart the kernel (Kernel > Restart) to clear everything.
- Loading multiple large datasets: Use generators or chunked reading. `pd.read_csv(path, chunksize=N)` returns an iterator instead of loading everything at once.

**Execution issues:**
- Notebook works for me but not others: Cells were run out of order. Fix: Kernel > Restart & Run All. If it fails, you have a hidden dependency on a deleted or reordered cell.
- Cell runs forever (hanging): The code might be waiting for input (`input()`), stuck in an infinite loop, or blocked on a network request. Interrupt with Kernel > Interrupt (or press `I` twice in command mode).
- Import errors after pip install: The package installed in a different Python than the kernel is using. Fix: run `!pip install package` inside the notebook, or check `!which python` matches your environment.

**Colab-specific:**
- Session disconnected: Free Colab times out after 90 minutes of inactivity. Save work to Google Drive or download files.
- GPU not available: Runtime > Change runtime type > select GPU. If all GPUs are busy, try again later or use Colab Pro.
- Files disappeared: Colab wipes the filesystem between sessions. Mount Google Drive for persistent storage: `from google.colab import drive; drive.mount('/content/drive')`.

Diagnostic steps:
1. What is the exact error message? (Check both the notebook and the terminal)
2. Does the issue happen after restarting the kernel and running all cells top to bottom?
3. How much data are you loading? (`df.info()` for dataframes, `tensor.shape` and `tensor.dtype` for tensors)
4. What environment are you using? (Local JupyterLab, VS Code, Colab)
5. Were packages installed in the same environment as the kernel? (`!which python` and `import sys; sys.executable`)

---

## Quick Reference: 常用 Jupyter 操作速查

> 这一节是 lesson 5 走完后沉淀的"操作清单"——前面诊断流程问"出啥错"的时候，你能直接对照下面这些看是不是操作本身没用对。

### 两种模式切换

| 键 | 动作 | 视觉提示 |
|---|---|---|
| `Esc` | 进**命令模式**（操作 cell） | 左边竖条变蓝、cell 边框蓝 |
| `Enter` | 进**编辑模式**（改 cell 内容） | 左边竖条变绿、cell 边框绿 |

**判自己在哪个模式**：看光标——闪烁的就是编辑模式，不闪的就是命令模式。

### 命令模式快捷键（最常用的 6 个）

| 键 | 动作 |
|---|---|
| `Shift+Enter` | 跑当前 cell + 跳到下一格（**神键，每天按 200 次**） |
| `A` | 在**上方**插一格（Above） |
| `B` | 在**下方**插一格（Below） |
| `M` | 把当前 cell 转成 **Markdown** |
| `Y` | 把当前 cell 转成 **Code** |
| `DD` | 连按两次 D，**删**掉当前 cell |

### 命令模式补充（按需查）

| 键 | 动作 |
|---|---|
| `Z` | 撤销上一次 cell 操作（删了能救回来） |
| `X` | 剪切当前 cell |
| `C` / `V` | 复制 / 粘贴 cell（在 cell 之间移动代码用） |
| `Ctrl+Shift+-` | 在编辑模式里**从光标处劈开**一个 cell |
| `Ctrl+Shift+H` | 弹所有快捷键的查表 |

### 编辑模式（写代码时用）

| 键 | 动作 |
|---|---|
| `Tab` | **自动补全**（变量名、函数名、属性） |
| `Shift+Tab` | 在函数调用的括号里按 → 弹**函数签名 / docstring** |
| `Ctrl+/` | 注释 / 取消注释当前行（或选中块） |
| `Ctrl+A` | 全选当前 cell |
| `Ctrl+Z` / `Ctrl+Shift+Z` | 撤销 / 重做 |

### 魔法命令（写在 cell 里）

| 命令 | 用途 | 例子 |
|---|---|---|
| `%timeit <expr>` | **微基准**：跑 N 次取最快 | `%timeit np.random.randn(10000)` |
| `%%time` | **整 cell 计时**：跑一次报 wall time | 写在 cell 第一行，整 cell 一起算 |
| `%matplotlib inline` | matplotlib 图**内联**到 cell 下面 | notebook 第一格 import 后写一次 |
| `%matplotlib widget` | 交互式图（缩放/悬停），要 `pip install ipympl` | JupyterLab 推荐 |
| `!shell_cmd` | 在 cell 里跑 shell | `!pip install pandas` |
| `%env VAR=value` | 查/设环境变量 | `%env CUDA_VISIBLE_DEVICES` |
| `%who` | 列**当前内存里所有变量名** | 排查"这变量哪来的" |
| `%whos` | 同上 + 类型 + 大小 | `%who` 的豪华版 |
| `%pwd` / `%cd` | 打印 / 切换工作目录 | 调试相对路径用 |
| `%load file.py` | 把外部脚本**整个塞进 cell** | 从 .py 搬到 notebook |
| `%run script.py` | 跑外部脚本（变量会进当前 kernel） | 在 notebook 里复用现成 .py |
| `%reset` | 清空 kernel 所有变量（要 `y` 确认） | **慎用**——`%reset -f` 不确认直接清 |

### Cell 显示规则（lesson 5 富输出那节）

| 写法 | 行为 |
|---|---|
| `df`（最后一行） | 自动显示 DataFrame 的富 HTML 表格 |
| `print(df)` | 强制纯文本（不推荐看大数据） |
| `display(df)` | 显式显示，**for 循环体里也能用** |
| `plt.plot(...)` + `%matplotlib inline` | 图直接出现在 cell 下面，**不需要 plt.show()** |
| `from IPython.display import Image; Image('x.png')` | 内嵌显示图片（最后一行） |
| `display(Image('x.png'))` | 同上，但**中间步骤**也能显示 |

### Kernel 常用操作

| 操作 | 在哪点 | 用途 |
|---|---|---|
| **Interrupt**（中断） | Kernel → Interrupt / 工具栏 ⏹ | 跑飞的 cell 立刻停 |
| **Restart**（重启） | Kernel → Restart | 内存清空、变量全没、import 全要重做 |
| **Restart & Run All** | Kernel → Restart & Run All | **最重要**：从头按顺序跑一遍，验证 notebook 是不是真的能复现 |
| **Restart & Clear Output** | 同菜单 | 重启 + 清掉所有 cell 输出（提交 PR 前用） |
| **Shut Down** | Kernel → Shut Down / File → Close and Shutdown | 彻底关掉 kernel，**释放内存**——不关 tab 直接关浏览器，kernel 还在后台跑 |

### 三个反直觉的小坑

1. **for 循环里的表达式不显示**：`for i in range(3): i` 跑完**啥也没有**。修法见 Addendum C。
2. **cell 顺序决定结果**：cell 5 改了 `x`、cell 3 再读 `x` 跟第一次跑不一样。修法：交作业前**永远 Restart & Run All**。
3. **相对路径跟启动终端的 cwd 走，不是 .ipynb 位置**：在 notebook 里 `Path("foo.png")` 跟 PowerShell 在哪个目录启动 JupyterLab 有关。修法见 Addendum B。

---

## 附录：英文正文中文翻译


### Kernel 崩溃

- **内存溢出（OOM）**：数据集或模型太大。修法：缩小 batch size、用 `pd.read_csv(path, chunksize=10000)` 分块读、用 `del variable` 释放然后 `gc.collect()`，或者换台内存大的机器。
- **原生库 segfault**：通常是 numpy / torch / tensorflow 与系统库版本不匹配。修法：重新建一个干净的 venv 重装。
- **Kernel 静默死亡**：去你启动 Jupyter 那个终端看真正的报错——notebook UI 经常把错误信息藏起来。

### 显示问题

- **图不显示**：在 notebook 最顶上加一行 `%matplotlib inline`。如果用 JupyterLab 想交互（缩放/悬停），改用 `%matplotlib widget`（需要先 `pip install ipympl`）。
- **DataFrame 出来是文本不是 HTML 表格**：确认它**是 cell 最后一个表达式**、**没被 `print()` 包住**。`print(df)` 给纯文本，光写 `df` 才给富表格。
- **图片不渲染**：
  ```python
  from IPython.display import Image, display
  display(Image(filename="path.png"))
  ```
- **Markdown 里 LaTeX 公式不渲染**：检查美元符号有没有成对。行内：`$x^2$`。块级：`$$\sum_{i=0}^n x_i$$`。

### 内存问题

- **notebook 占内存太大**：变量在 cell 之间是活的，先跑 `%who` 列出现在内存里所有变量，大对象用 `del var_name` 释放，然后 `import gc; gc.collect()`。
- **内存越用越大**：大概率是把大变量反复重赋值、旧引用没释放。直接 **Kernel → Restart** 清空重来。
- **连续加载多个大文件**：用生成器或分块读。`pd.read_csv(path, chunksize=N)` 返回迭代器，不会一次性把整个文件吃进内存。

### 执行问题

- **本地能跑、别人跑挂**：cell 顺序乱了。修法：**Kernel → Restart & Run All** 从头按顺序跑一遍。如果某格挂掉，说明那格藏了对已删除 / 已重排 cell 的依赖。
- **Cell 跑个不停（卡死）**：代码可能在等 `input()` 输入、卡在死循环、或者阻塞在网络请求。按 **Kernel → Interrupt** 中断（命令模式下按两次 `I` 也行）。
- **`!pip install` 之后还是 ImportError**：包装到了**别的 Python** 里，不是 kernel 正在用的那个。修法：在 notebook 里跑 `!pip install package`，或者用 `!which python` 和 `import sys; sys.executable` 确认 kernel 跟 pip 装的是同一个解释器。

### Colab 专属

- **会话断线**：免费版 Colab 闲置 90 分钟后会自动断。重要文件存到 Google Drive 或者直接下载。
- **GPU 不可用**：**Runtime → Change runtime type → 选 GPU**。所有免费 GPU 都被占满时，要么等、要么上 Colab Pro。
- **文件不见了**：Colab 每次新会话都会清空虚拟机磁盘。挂载 Google Drive 做持久化：
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### 诊断流程

遇到 notebook 故障，按这个顺序问：

1. **完整的报错信息是什么？**（notebook 和启动它的终端都要看）
2. **Restart & Run All 之后还复现吗？** 不复现 → 顺序错乱；复现 → 继续往下查
3. **你加载了多大的数据？** DataFrame 用 `df.info()`，tensor 用 `tensor.shape` + `tensor.dtype`
4. **跑在哪个环境？** 本地 JupyterLab / VS Code / Colab
5. **包装在哪个 Python 里？** `!which python` + `import sys; sys.executable`，确认和 kernel 用的解释器一致

---

## Addendum: 实战补遗（lesson 5 当天踩过的坑）

> 这一节是 lesson 5 实操后补进来的，不在原课程 artifact 范围，记录那种"原文档没写、但真上手就会撞上"的问题。

### A. venv 是 uv 管的 → 没 pip，装包走 `uv pip install`

**症状**：`python -m pip install pandas` 报 `No module named pip`。

**原因**：venv 是用 `uv venv` 创建的（或者装了 `uv` 之后由它管理），uv 默认不往 venv 里塞 pip。

**修法**：用 uv 直接装，落到项目 venv 里：

```powershell
uv pip install pandas --python C:\python_code\ai-engineering-from-scratch\.venv\Scripts\python.exe
```

或者用 `uv run` 自动管理临时环境（适合单次跑、不污染 venv）：

```powershell
uv run python your_script.py
```

**判断自己 venv 是不是 uv 管的**：看 `.venv/Scripts/` 里有没有 `uv.exe` / `uvx.exe`，或者有没有 `pyproject.toml` / `uv.lock`。

### B. 相对路径在 notebook 和 script 里行为不同

- **script** 里写 `Path("foo/bar.png")`，相对的是"你 `python xxx.py` 时所在的当前目录"。
- **notebook** 里写 `Path("foo/bar.png")`，相对的是"**启动 JupyterLab 那个终端**的当前目录"，不是 .ipynb 文件所在目录。

`os.chdir()` 在 notebook 里**改不了 kernel 启动时锁定的 cwd**（在某些前端/配置下能改，行为不一致）。

**最稳的修法**：

```python
from pathlib import Path
HERE = Path.cwd()       # notebook 里查 cwd 到底是哪
print(HERE)             # 先打一下，确认跟预期一致
save_path = HERE / "out.png"
```

script 里用 `Path(__file__).parent` 锚定自身位置，跟 cwd 解耦；notebook 里 `__file__` 行为怪，优先 `Path.cwd()`。

### C. for 循环体里的表达式**不会**自动显示

```python
for i in range(3):
    i
```

跑完**啥也没有**——`for` 语句本身就是 cell 最后一个表达式（值 `None`），`for` 体里的 `i` 没被自动显示机制捕获。

修法：把结果收集起来再显示：

```python
results = [i for i in range(3)]
results   # 现在是 cell 最后一个表达式，会显示
```

**调试时临时看每步**：用 `display()` 或者 `print()` 显式打，别指望"循环里写表达式就能看到"。

