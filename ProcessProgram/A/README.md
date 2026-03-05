# ProcessData/Program 说明

这个目录里的文件主要用于：
- 从 `OriginalData` 的 256×256 原始 ToT/ToA 文本矩阵中提取/清洗出可用于训练的样本（如 100×100 轨迹矩阵）。
- 把 ToT 文本矩阵渲染成 PNG 便于肉眼检查。
- 把不同来源/类别/模态的数据按目录规则合并成统一数据集。
- 做一些基础统计与分布对比，验证数据可分性。

> 注：这些脚本/Notebook 写死了一些绝对路径示例（如 `d:\Project\Timepix\...`）。复现时优先改成你当前机器的路径，或改成相对路径。

---

## 文件一览（做什么 / 输入 / 输出）

### 1) DataExploration.ipynb —— 数据探索与统计

**用途**
- 对 `ProcessData/Alpha` 与 `ProcessData/Electron` 下的 ToT/ToA 文本矩阵做快速探索。
- 统计每个文件的总和、非零占比、非零均值/方差等，并按粒子类型、角度、模态聚合。
- 画 ToT “总沉积能量（sum）” 的直方图等，用于粗略判断分布差异。

**输入（默认逻辑）**
- `ProcessData/Alpha/<angle>/{ToT,ToA}/*.txt`
- `ProcessData/Electron/<angle>/{ToT,ToA}/*.txt`

**输出**
- Notebook 内的表格/图（不写文件）。

**关键点**
- `load_detector_matrix()` 允许 `(100,100)` 或 `(256,256)`。
- `collect_particle_stats()` 会遍历所有角度目录并收集每个文件的统计。

---

### 2) TrajectoryExtraction.ipynb —— 从 256×256 提取轨迹并保存为 100×100

**用途**
- 从原始 256×256 的 ToT/ToA 矩阵中，找到连通的激活像素区域（8 邻域连通域）。
- 通过几何规则过滤掉噪声/无效轨迹。
- 将每条有效轨迹平移到 100×100 画布中心（按质心对齐），分别生成 ToT/ToA 的 100×100 文本矩阵。

**输入（你需要在第一段改路径）**
- `INPUT_DIR = OriginalData/.../<angle>/`（例：`OriginalData/Alpha/60`）
- 目录下的文件名需要能配对：同一事件的 ToT 与 ToA
  - 通过 stem 中是否包含 `_ToT` / `_ToA` 来配对

**输出（你需要在第一段改路径）**
- `OUTPUT_TOT_DIR = ProcessData/Alpha/<angle>/ToT/`
- `OUTPUT_TOA_DIR = ProcessData/Alpha/<angle>/ToA/`
- 输出文件名规则：`{base_stem}_{channel}_{index:04d}.txt`
  - 例：`1_r0000_ToT_0001.txt`、`1_r0000_ToA_0001.txt`

**过滤规则（Notebook 中可调）**
- `MIN_PIXEL_COUNT`：连通域像素数下限
- `MAX_AXIS_RATIO`：短轴/长轴的比值阈值（越小越“细长”）
- `MIN_LONGEST_SPAN`：外接矩形长边下限
- 触边（`touches_edge`）的一律丢弃

**ToA 处理**
- `TOA_SCALE`（默认 10000）把 ToA 浮点数缩放成整数保存（避免小数）。

---

### 3) ToT_to_Image.ipynb —— 批量把 ToT 文本矩阵渲染为 PNG

**用途**
- 递归搜索某个目录下所有包含 `ToT` 关键字的 `.txt` 矩阵文件。
- 将 0 像素显示为白色，>0 显示为黄色系渐变（值越大颜色越深），并可放大（保持像素格感）。
- 适合作为 sanity check：快速查看样本形态、噪声、空事件等。

**输入（在“参数设置”单元改）**
- `in_dir`：输入根目录（会 `rglob("*ToT*.txt")`）
- 支持 256×256；若遇到非 256×256 会告警但仍处理。
- 对 `100×100` 的数据可选“中心裁剪成 32×32”（便于看局部）：`crop_for_100=True`。

**输出**
- `out_dir`：输出根目录
- 保持相对层级：输出 PNG 的相对路径与输入 `.txt` 一致，仅后缀改为 `.png`
- 生成日志：`out_dir/render_log.json`（记录成功/失败与错误信息）

---

### 4) merge_modalities_by_categories.py —— 跨多个数据源按类别合并，同时保留 ToA/ToT

**用途**
- 从多个“数据集根目录”中，挑选若干类别（category 文件夹）合并到一个 target。
- target 中仍按模态分目录（例如 `ToA/<category>/...`、`ToT/<category>/...`）。
- 解决文件重名冲突：跳过/覆盖/重命名（默认重命名 `_1/_2/...`），也可给文件名前加数据源名。

**输入目录布局（两种都支持）**
- `modality_first`（默认）：`source/<modality>/<category>/...`
- `category_first`：`source/<category>/<modality>/...`

**输出目录结构（固定）**
- `target/<modality>/<category>/...`

**典型命令**
- 预览（不复制）：
  - `python merge_modalities_by_categories.py --sources ... --categories ... --target ... --dry-run`
- 真正复制并在冲突时重命名：
  - `python merge_modalities_by_categories.py --sources ... --categories ... --target ... --on-conflict rename --prefix-source-name`

---

### 5) merge_alpha_0_1.py —— 合并 AlphaAnalysis 里 class 0/1（按 ToA/ToT 分开）

**用途**
- 这是一个更“定制化”的小工具：把 `AlphaAnalysis/data/Alpha/0` 与 `AlphaAnalysis/data/Alpha/1` 合并到 `AlphaAnalysis/data/Alpha/0_1_merged`。
- 为避免重名，目标文件会加上类别前缀：`0_<原名>`、`1_<原名>`。

**输入（默认）**
- `AlphaAnalysis/data/Alpha/0/{ToA,ToT}`
- `AlphaAnalysis/data/Alpha/1/{ToA,ToT}`

**输出（默认）**
- `AlphaAnalysis/data/Alpha/0_1_merged/{ToA,ToT}`

**运行方式**
- 默认是 dry-run（只给统计，不拷贝）：`python merge_alpha_0_1.py`
- 真正执行：`python merge_alpha_0_1.py --run`
- 允许覆盖：`python merge_alpha_0_1.py --run --overwrite`

---

## 推荐工作流（从原始数据到可训练数据）

1) 用 `TrajectoryExtraction.ipynb`：`OriginalData/...` → `ProcessData/.../ToT|ToA` 的 100×100 轨迹矩阵。
2) 用 `DataExploration.ipynb`：对 `ProcessData` 做统计/分布对比，确认数据质量与可分性。
3) 用 `ToT_to_Image.ipynb`：把部分 ToT 样本渲染成 PNG 做可视化抽检。
4) 用 `merge_modalities_by_categories.py` 或 `merge_alpha_0_1.py`：按实验设计合并类别/来源，生成最终训练集目录结构。
