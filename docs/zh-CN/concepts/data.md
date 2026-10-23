[English](../../concepts/data.md)

<a id="data-pipeline"></a>
# 数据流水线

`kohakumoss_data` 将数据集转换为打包批次。它唯一的第一方依赖是
`kohakumoss_common`：rank 和 world 都是普通整数，编码器是普通可调用对象，
模型侧的布局（`SeqInfo`）由训练器根据 `hw` 和 `text_len` 构建。

```
Source.walk() ──▶ catalog file (cache) ──▶ Catalog (columns) ──▶ Planner(seed, rank,
                                                                  world) ──▶ BatchPlan
                                                         │
                     MossDataset.materialize: rows fetched, captions, text dropout,
                     cache lookups, decode ──▶ HostBatch (pinned) ──▶ trainer
```

操作指南（布局、缓存脚本、导入分片）见 [datasets.md](../guides/datasets.md)。
本页按顺序介绍：数据集条目与数据源、描述文本、分辨率与批次划分、缓存，
然后是目录、批次、加载器与文件格式。

<a id="dataset-entries"></a>
## 数据集条目

每个 `DATASETS` 项都是 `{"source": name, **source arguments}`，再加上数据集级别的键，
这些键不会传给数据源：

| 键 | 效果 |
| --- | --- |
| `weight` | 乘以该条目中每一行的权重（用于混合数据集） |
| `caption` | 该条目的描述文本模板（默认：`CAPTION`） |
| `loss_weight` | 该条目各行的逐样本损失权重（`HostBatch.weight`） |
| `catalog` | 预先构建的目录文件，存在时读取它而不是扫描 |
| `import_to` | 仅限 `catalog_build.py`：将图像复制到一个 `moss_shard` 目录 |

<a id="sources"></a>
## 数据源

| 数据源 | 布局 | 参数与字段 |
| --- | --- | --- |
| `kohya` | `<repeats>_<name>/img.ext` + `.txt` | `repeat_prefix`：`first`（开头的整数，kohya）、`product`（所有开头整数的乘积）、`ignore`；`artist_from_folder`：`None`、`"name"`（不含数字前缀的文件夹名）或正则表达式；`include_root`（默认 False）：也包含直接位于 `path` 下的图像，重复次数为 1、无画师；未找到图像时会报错，指出布局和 `include_root` |
| `folder` | 任意目录树，伴随文件 `.txt` / `.caption` / `.json` | `txt_as`：`tags` 或 `nl`；`recursive`（默认 True）；json 键 `tags`、`caption`、`artist`、`rating`，其余保留在 `row.json` 中 |
| `jsonl` | 每行一条记录，图像键为 `image` / `file` / `path` / `file_name` | `root`（图像路径相对于它；默认为文件所在文件夹）；记录键 `tags`（列表或字符串）、`caption` / `text` / `caption_nl`、`artist`、`rating`、`weight`、`group`，其余保留在 `row.json` 中 |
| `parquet` | 路径列（`image_col`，默认 `image`）或字节列（`bytes_col`） | `root` 与 `jsonl` 相同；列与 jsonl 的键相同；需要 `pyarrow`（可选依赖 `parquet`） |
| `webdataset` | 未压缩的 `.tar`（`path` 是 glob），成员为 `<key>.<ext>` | `<key>.txt` 使用 `txt_as`；`<key>.json` 使用 folder 的 json 键；加载时按字节偏移读取成员 |
| `moss_shard` | 原生格式 | 参见 [`moss_shard` 格式](#the-moss_shard-format) |

- 行以图像文件为标识；没有对应图像的伴随文件会被忽略。
- 扫描只读取图像头（尺寸、EXIF 方向、alpha）。
- **内容标识**（缓存键的依据）：

  | 标识 | 使用条件 |
  | --- | --- |
  | 文件指纹：绝对路径 + 大小 + mtime | `kohya`、`folder`、`jsonl`、`parquet`（路径列）的默认方式 |
  | 图像字节的 blake2b | 上述数据源设置 `hash_content=True` 时；`webdataset` 和 parquet 字节列始终使用；导入分片的数据 |

  路径使大小和 mtime 相同的两个文件（以秒级时间戳解压的归档、`cp -p` 副本）
  具有不同标识；它也将以指纹为键的缓存条目绑定到文件位置，
  因此移动数据集会重新编码，而内容哈希不依赖位置。

<a id="captions"></a>
## 描述文本

模板（`CAPTION = {"template": ...}`）：

- `tags`：先保留字段（`keep`，如 `artist`），然后是 `keep_first` 个固定标签，
  最后是其他标签。`|||` 标签结束固定部分；`%%` 标签用于分组。
  每组按 `group_dropout` 丢弃，每个标签按 `tag_dropout` 丢弃；
  启用 `shuffle` 时打乱组和标签。`underscores` 将 `_` 转为空格，
  `unescape` 将 `\(` 转为 `(`。
- `nl`：使用 `caption_nl`，将保留字段以 `a, b. ` 形式加在前面，缺失时回退到标签。
- `mix`：以概率 `p_nl` 使用自然语言，否则使用标签。
- `json`：使用 `row.json` 加上列出的行字段，以 `key: value` 行或紧凑 JSON 形式输出，
  应用 `field_dropout`（`keep` 中的字段从不丢弃）。

行 `r` 的描述文本变体 `k` 由一个以 `(seed, r, k)` 为种子的生成器渲染，
因此每行恰有 `max(1, CAPTION_VARIANTS)` 个描述文本（0 表示抽取变体 0）：
文本缓存保存的内容就是训练抽取的内容，因为缓存处理从规划器获取这一数量。

文本 dropout（`TEXT_DROP`）在已知长度之前，按 (seed, index, rank, slot) 决定：
被丢弃样本的描述文本为 `""`，其嵌入为缓存的空提示词。

<a id="resolution-batching-over-long-samples"></a>
## 分辨率、批次划分与超长样本

四个独立选择：目标尺寸、批次划分、文本词元预算和超长样本处理。
所有尺寸都是 `SIZE_STEP` 的倍数（16 = VAE /8 x patchify 2）；
批次中的 `hw` 是潜变量网格（像素 / 16）。

<a id="target-size-resolution_mode"></a>
### 目标尺寸：`RESOLUTION_MODE`

| 模式 | 目标尺寸 |
| --- | --- |
| `resize`（默认） | 原生宽高比，面积 ~= `TARGET_AREA`：每个样本的图像词元数几乎相同 |
| `bucket` | 最接近的对数宽高比桶，面积约为 `TARGET_AREA`；使用 `UPSCALE="never"` 时，较小图像保留原生尺寸，作为自身的桶；批次绝不混合不同桶 |
| `square_crop` | 边长 sqrt(`TARGET_AREA`)，中心裁剪 |
| `shifted_square` | 同上，裁剪位置由种子决定 |

- 尺寸为 (h, w) 像素；裁剪框为源图像像素中的 (left, top, right, bottom)，
  具有目标宽高比，先裁剪再缩放到目标尺寸。
- `ASPECT_RANGE` 限制 w/h（范围外的图像会裁剪到边界）。
- `UPSCALE="never"` 让小于目标的图像保留原生尺寸。
- `CROP`（`DataConfig.crop`）为 `center`（默认）或 `random`；离线潜变量缓存需要 `center`。

<a id="batching-batching"></a>
### 批次划分：`BATCHING`

| 模式 | 规则 |
| --- | --- |
| `tokens`（默认） | 遍历由种子确定的样本顺序，在批次保持 <= `MAX_TOKENS` 时不断添加样本（图像 + 文本词元）；会超出预算的样本作为下一批次的起点 |
| `tokens_ffd` | 对每个包含 256 个样本的窗口按大小降序首次适应算法装入容量为 `MAX_TOKENS` 的桶（批次更密集，顺序打乱） |
| `count` | 连续的 `BATCH_SIZE` 个样本（末尾不完整批次被丢弃） |

- `MAX_SAMPLES` 额外限制每个词元批次的样本数（0 = 不限制；此时上限会设为
  高于任何批次大小，因此每个批次划分函数都会无条件应用它）。
- `bucket` 对每个桶运行一次批次划分函数；`bucket` 和 `tokens_ffd`
  的批次在划分后会被打乱。

<a id="text-tokens-in-planning-token_budget"></a>
### 规划时的文本词元：`TOKEN_BUDGET`

| 值 | 文本词元 |
| --- | --- |
| `estimate`（默认） | 根据目录的字符列：数据集描述文本模板估算每行不应用 dropout 时的描述文本长度（`estimate_chars`），再计算 `ceil(chars / TEXT_CHARS_PER_TOKEN) + TEXT_OVERHEAD_TOKENS`，上限为 `MAX_TEXT_TOKENS`；不读取任何行 |
| `render` | 渲染每个抽到的描述文本（获取行）：已缓存时使用离线文本缓存长度，否则对渲染后的字符串使用相同估算方式 |
| `exact` | 从离线文本缓存获取每个描述文本的长度，因此 `MAX_TOKENS` 是硬上限；静态流水线微批次形状（`MAX_TOKENS`、`MAX_SAMPLES`）需要此模式（`kohakumoss_data.pipeline_problems` 会列出缺失项） |

<a id="over-long-samples-overlong"></a>
### 超长样本：`OVERLONG`

单个样本超过 `MAX_TOKENS` 时的处理方式：

| 模式 | 效果 |
| --- | --- |
| `drop`（默认） | 跳过 |
| `truncate_text` | 只保留前 `MAX_TOKENS - image tokens` 个文本词元。文本编码器具有因果性，因此缓存嵌入的前缀恰好就是描述文本前缀的嵌入；聊天模板末尾（`<|im_end|>` / assistant 轮次）也随之被截去 |
| `drop_tokens` | 保留 `MAX_TOKENS - text tokens` 个图像词元，由带种子的抽样选择；`img_keep[i]` 保存它们在行优先网格中的已排序索引，`latents` 只包含它们，而 `build_seqinfo(hw, text_len, img_keep=...)` 赋予每个保留词元其原始网格位置 |

- 调整后的样本至少保留一个文本词元和一个图像词元；否则丢弃。
- 在线编码时，加载器返回完整图像（`pixels`）、完整提示词，以及
  `img_keep` / `text_limit`；训练器先编码，再收集 / 截断。
- 在 `count` 批次划分下，`MAX_TOKENS` 也逐样本生效。

<a id="data-knobs"></a>
### 数据配置项

除 `latent_factor` 外，每个 `DataConfig` 字段都是训练脚本中同名的大写配置项
（`DataConfig.from_knobs`），包括 `CROP`、`PLANNER_KWARGS`（`window`、
`drop_last`、……；这些会覆盖根据其他配置项推导出的规划器参数）、
`LATENT_ENCODER_ID`、`TEXT_ENCODER_ID`、`LATENT_ENCODER_SPEC` / `TEXT_ENCODER_SPEC`
（由训练器和缓存处理设置，参见[缓存键](#cache-keys)）、`LATENT_DRAW`
（也用于在线编码时的 VAE 抽样）、`BACKGROUND`、`RESAMPLE`、
`TEXT_CHARS_PER_TOKEN`、`TEXT_OVERHEAD_TOKENS`、`MAX_TEXT_TOKENS`、`PREFETCH` 和
`PIN_MEMORY`。`kohakumoss_data.data_problems(cfg)` 会列出所有会使
`MossDataset(cfg)` 失败的值。

<a id="the-plan"></a>
### 计划

- 每个 epoch 都是加权行多重集的带种子排列（行 i 出现 floor(w) 次，
  再以概率 frac(w) 多出现一次）；每次出现都会获得由种子确定的
  裁剪 / 翻转 / 描述文本变体 / 保留抽样。
- 规划时的文本词元遵循 `TOKEN_BUDGET`（见上文）；在线缓存在运行过程中增长，
  因此规划从不读取它：计划是数据、种子和配置项的纯函数，
  重启或恢复会将每个索引映射到相同样本。
- 带删除标记的行权重为 0，永远不纳入计划。
- 全局批次 `g = index * world + rank`，因此各 rank 不会共享批次；
  通过遍历 epoch 长度定位 `g`，规划器保留最近五个 epoch。

<a id="caches"></a>
## 缓存

`LATENT_CACHE` 和 `TEXT_CACHE` 为 `off | offline | online`；两者都位于 `CACHE_DIR`
（`latents.db`、`text.db`，KohakuVault）。空的 `CACHE_DIR` 会解析为第一个带有
`path` 的数据集的 `<path>/cache`（当路径是文件或 glob 时，使用其所在文件夹）
（`kohakumoss_data.resolved_cache_dir`）；训练器和缓存脚本使用同一规则。

| 模式 | 未命中时 |
| --- | --- |
| `offline` | 抛出 `CacheMiss`（先运行缓存脚本） |
| `online` | 返回像素 / 提示词及 `latent_keys` / `text_keys`，以便训练器编码并回写结果 |
| `off` | 始终返回像素 / 提示词 |

存储数据类型：`bf16`、`fp16`、`fp32` 或 `fp8`（e4m3，带有逐张量
fp32 缩放 `absmax / 448`，最大相对误差约为 7%，字节数为 bf16 的一半）。

缓存命中返回已存储（已舍入）的值。`LatentCache.as_cached` /
`TextCache.as_cached` 在新编码所在的设备上，以完全相同的方式进行舍入，
因此在线运行中，无论样本命中还是未命中，训练使用的值都相同。

<a id="cache-keys"></a>
### 缓存键

- 潜变量：内容标识 x 编码器 x 存储数据类型 x 目标尺寸 x 裁剪框 x
  翻转 x 背景（仅 alpha 图像）x 重采样滤波器 x VAE 抽样
  （`latent_draw`，`mode` 或 `sample`）
- 文本：渲染后的描述文本 x 编码器 x 存储数据类型

编码器部分是编码器 id 加上其计算规格（`LATENT_ENCODER_SPEC`、
`TEXT_ENCODER_SPEC`；`kohakumoss_data.encoder_key_id`）：

- VAE 的计算数据类型（`fp32` / `bf16` / `fp16`）；
- 文本编码器的 `<dtype>/<precision>`，再加上仅权重量化 int8 / fp8 基座的量化组，
  或 fp8 内核方案（`kohakumoss.encoders.specs`）。

fp16 和 bf16 的 VAE，或者 int8 和 16 位的文本编码器，会生成不同的编码，
因此按一种方式构建的缓存绝不会用于以另一种方式计算的运行：
该运行会未命中，未命中消息会指出不同的规格。
放置方式（内存模式、批次划分、布局、内核）不在键中。
缓存处理从其构建的编码器（`encoder_spec`）获取规格，并拒绝规格不同的数据集；
训练器从自身的编码器配置项推导规格（`VAE_DTYPE`、在其设备上解析的 `TEXT_ENCODE`；
`training.build.model.data_knobs`）。

`MAX_TOKENS` 不是影响键的配置项：无论批次预算如何，缓存都保存完整网格潜变量和完整描述文本。

<a id="offline-precompute"></a>
### 离线预计算

`scripts/data/cache_latents.py` / `cache_text.py` 为 `offline` 填充
`latents.db` / `text.db`（tests/data/cache/test_cache.py）。
潜变量处理在主线程读取每个分块的图像字节，并在 `NUM_WORKERS` 个线程上解码，
比编码器提前一个分块。

文本缓存的长度通过打包长度表读取（`idx:lengths`，每条目 20 字节，使用二分查找），
该表在文本缓存处理结束时写入；没有该表的 vault 会在启动时扫描一次。

<a id="online-write-back"></a>
### 在线回写

`online` 运行通过 `CacheWriter` 回写未命中项（tests/data/cache/test_writer.py），
因此后续 epoch 或运行会命中。在线潜变量由 VAE 在其 `VAE_ENCODE` 放置方式下生成，
每次 VAE 处理 `ahead` 个主机批次（[offload.md](../internals/offload.md#vae)）。

- `CacheWriter` 是训练器的 `on_encoded` 钩子：它在旁路流上将每个批次的新编码
  复制到锁页主机内存（在计算流工作完成后，每批次一个事件），
  将主机张量放入队列，然后后台线程等待事件并写入；队列保存主机张量，
  所有复制都在旁路流上运行。
- 只写入完整值：完整网格潜变量（保留词元的子集不是其键对应的值）
  和未被 `text_limit` 截断的文本。
- 每个 rank 只写自己的分片，即主文件旁的 `latents.r<rank>.kv` / `text.r<rank>.kv`，
  因此任何文件都不会有两个写入者。每次查找（`get`、`in`、长度）
  都读取主文件与现有分片的并集（重新列出分片的频率不高于每 5 s 一次）。
- `scripts/data/cache_merge.py`（`kohakumoss_data.cache.store.merge_shards`，
  单进程，且没有运行正在写入）将分片合入主文件并删除分片。

<a id="cache-misses"></a>
### 缓存未命中

每次缓存处理和每个 `online` 运行都会将影响键的配置项记录到
`<cache_dir>/latents.settings.json` / `text.settings.json`
（每种不同构建一条记录；`kohakumoss_data.cache.settings.FIELDS`）。
`offline` 未命中会抛出 `CacheMiss`，它会：

- 指出第一个缺失样本，以及与最接近的已记录构建不同的键配置项
  （缓存没有记录时，列出该运行的所有键配置项）；
- 以数据配置行形式打印键配置项，并给出将每个键配置项通过 `--set`
  传入的缓存处理命令（Windows 使用 PowerShell 引号规则，其他平台使用 POSIX），
  以便重建与运行匹配（tests/data/cache/test_cache_miss.py）。

如果缓存所记录的构建缺少某个键配置项，说明它是在其他键下写入的，
因此每个条目都会未命中；消息会说明这一点，并提供相同的重建命令。

<a id="the-catalog"></a>
## 目录

`Catalog` 将规划用的列保存在 numpy 中（`w`、`h` 为 int64，`weight` 为 float32，
`has_alpha` 和 `live` 为 bool，`group_id` 为指向 `groups` 的 int64，
描述文本来源的字符计数 `chars[...]`，`source` 为 int64）。

- `catalog[i]` 根据各列和该行的可变字段（locator、tags、caption_nl、artist、
  rating、json、extra）构建 `CatalogRow`，通过每进程 LRU（4096 行）按需获取。
- 规划器、潜变量缓存处理、统计和 `Catalog.concat` 只读取列；
  在批次物化或描述文本渲染时才获取行。
- `Catalog(rows)` 从内存中的行构建相同对象。

<a id="the-catalog-cache"></a>
### 目录缓存

扫描型数据源（`kohya`、`folder`、`jsonl`、`parquet`、`webdataset`）
列出遍历项的开销很低，而探测开销较高。一个遍历项有一个稳定键、
一个含 8 个整数的戳，以及一个探测操作：

| 数据源 | 遍历项 | 戳 |
| --- | --- | --- |
| `kohya`, `folder` | 一张图像 | 图像及其 `.txt`、`.caption`、`.json` 的 (size, mtime_ns)（缺失时为 -1），每个目录调用一次 `os.scandir` 获取 |
| `jsonl`, `parquet`（路径列） | 一条记录，键为 `locator#n` | 图像的 (size, mtime_ns) 加上记录的 63 位哈希 |
| `parquet`（字节列） | 该文件 | 其 (size, mtime_ns) |
| `webdataset` | 一个 tar | 其索引、size、mtime_ns |

每个数据集条目都有一个目录文件 `<cache>/catalog/<source>-<digest>.db`
（摘要包含数据源种类、其绝对路径及其选项）；`<cache>` 是潜变量 / 文本缓存目录，
或这些缓存关闭时原本会使用的目录。

启动时，写入者遍历数据源并比较戳：

| 遍历项 | 操作 |
| --- | --- |
| 新增 | 探测并追加 |
| 发生变化 | 重新探测并原地重写（id 相同） |
| 遍历未找到 | 加删除标记（`live` 为 0，权重为 0；其 id 永远不会重用于其他项） |

冷扫描按遍历顺序追加，因此其 id 与未缓存扫描相同。
探测在 `DATA_SCAN_WORKERS` 个 spawn 进程中运行并返回行；
只有写入者接触文件，每 4096 行提交一次。
通过一行日志报告 `catalog: N images (M from cache, K probed) in X s`。

| `DATA_RESCAN` | 扫描方式 |
| --- | --- |
| `False`（默认） | 增量 |
| `True` | 原地重新探测每一项 |
| `"never"` | 不遍历，直接打开文件：用于已知不会变化的大数据集 |

`scripts/data/catalog_cache.py` 显示、删除或压缩这些文件（`ACTION`、
`kohakumoss_data.scan.manage`）；压缩以新的 id 按遍历顺序重写有效行，
复制未变化的行，其结果与冷扫描写出的内容完全相同。

**写入者协调。**

- 写入者获取 `<file>.lock`（O_CREAT | O_EXCL，保存其 pid 和主机；
  如果锁中 pid 在本主机上已不存在，则锁已失效），清除 `<file>.ready`，
  更新文件，写入 `<file>.ready`（代次、行数），然后释放锁。
- 读取者（`MossDataset(cfg, writer=False)`，或发现有效锁的写入者）
  最多等待 `DATA_WAIT` 秒，直到不存在有效锁且标记存在，然后打开文件。
- 无法读取的锁文件（写入者尚未写入内容）在 60 s 后视为失效；
  来自另一主机的锁永远不会被视为失效。
- 训练器先在 rank 0 上构建数据集，然后其他 rank 在屏障后以读取者身份构建。

<a id="the-catalog-file"></a>
### 目录文件

一个 KohakuVault 文件，同一路径上包含一个 ColumnVault 和一个 KVault：

| 部分 | 内容 |
| --- | --- |
| 固定 i64 列 | `w`, `h`, `has_alpha`, `live`, `group_id`, `tags_chars`, `nl_chars`, `artist_chars`, `rating_chars`, `json_chars`；戳 `st_size`, `st_mtime`, `s1_size` ... `s3_mtime` |
| `weight32` | 每行 4 个小端 float32 字节 |
| `item`（str） | 每行来源的扫描项 |
| KVault `r<id, 8-byte BE>` | 该行可变字段的 msgpack |
| KVault `meta:info`, `meta:groups` | 模式、代次、行数、数据源种类；组名 |

打开时，对每个规划列执行一次范围读取，直接读入 numpy；戳和遍历项仅在重新扫描时读取。
`moss_shard` 数据集的 `catalog.db` 和 `catalog_build.py` 生成的文件使用相同布局，
这是运行时读取的唯一布局（其他布局文件参见
[datasets.md](../guides/datasets.md#4-optional-import-into-shards)）。
`moss_shard` 数据集缺少其 `shards/` 时会报错并说明原因。

<a id="the-moss_shard-format"></a>
## `moss_shard` 格式

```
dataset/
  catalog.db          a catalog file (layout above); each row's extra holds the
                      blake2b content hash and the source locator (`origin`)
  shards/00000.bin    concatenated encoded images (rolled at shard_bytes, 1 GiB default)
  shards/00000.idx    32-byte records: offset u64, size u32, w u16, h u16, codec u8,
                      flags u8 (bit 0 = alpha), 6 pad, content-hash prefix u64
```

`scripts/data/catalog_build.py` 导入任意数据源（`import_to`），或只写出其目录
（`catalog`），以便后续运行跳过扫描。

<a id="hostbatch"></a>
## HostBatch

`MossDataset.materialize(index, rank, world)` 返回一个 `HostBatch`；
图像词元以样本优先排列，每个样本的潜变量网格内按行优先排列。

| 字段 | 内容 |
| --- | --- |
| `ids` | int64 [B]，目录行 id |
| `hw` | int32 [B, 2]，潜变量网格（像素 / 16） |
| `text_len` | int32 [B]，文本词元数；设置 `text` 时为最终值，否则由消费方在编码 `prompts` 后填入 |
| `text_drop` | bool [B]，文本 dropout 抽样结果 |
| `weight` | float32 [B]，各条目的 `loss_weight` |
| `img_keep` | 每个样本在 `drop_tokens` 后保留的已排序网格索引，或 None（全部保留） |
| `latents` | 缓存的打包潜变量 [N_img, C]，只包含保留的词元 |
| `text` | 缓存的打包文本嵌入，已截断到 `text_limit` |
| `pixels` | 潜变量未缓存时的完整图像，uint8 [H, W, 3]；消费方先编码，再收集 `img_keep` |
| `prompts` | 文本未缓存时的描述文本；消费方在编码后保留前 `text_limit` 行 |
| `text_limit` | 逐样本，> 0 时限制文本词元数（`truncate_text`） |
| `latent_keys` / `text_keys` | 每个样本的缓存键（相应缓存关闭时为空），用于在线回写 |
| `index` / `rank` | 计划坐标 |
| `layout` | 消费方的打包序列布局，在文本长度为最终值时由加载器工作进程中的变换构建（否则为 None） |

`HostBatch.pin()`（DataLoader 的锁页钩子）会对所有张量字段、列表字段（如 `img_keep`）里的张量，以及具有自身 `pin_memory()` 方法的 `layout` 执行锁页处理；训练器使用的 `SeqInfo` 也实现了该方法。工作进程通过共享内存交付各张量，只要相关共享存储仍存活，主进程就会为它保留一个打开的文件描述符。没有锁页的字段在批次被持有期间会一直占用文件描述符，而训练器在第一次前向传播前就会持有完整的 `GRAD_ACC` 批次组。若 `SeqInfo` 没有锁页，每批次约需 16 个打开文件；`GRAD_ACC=64` 会超过 Linux 默认的 1024 文件描述符上限。锁页后，被持有的批次不再保持打开文件。

<a id="loader"></a>
## 加载器

`Loader` 在 `BatchStream` 之上封装一个 `DataLoader(batch_size=None)`：
`W` 个工作进程中的工作进程 `w` 物化索引 `start + w, start + w + W, ...`，
DataLoader 按顺序轮流获取各工作进程的数据，因此无论工作进程数是多少，数据流都相同。
恢复状态为 `{seed, start}`。

- **全新的工作进程。** 工作进程从全新解释器启动
  （`kohakumoss_data.loader.stateless.worker_context`：Linux 上为 `forkserver`，
  Windows 上为 `spawn`），因此数据集（数据源、目录、模板、缓存）按值 pickle；
  文件句柄在每个进程中延迟重新打开。批次不依赖启动方式。
- **变换。** `Loader(transform=...)` / `build_loader(cfg, rank, world,
  transform=...)`（可 pickle 的可调用对象或点分路径）在工作进程内应用于每个批次，
  然后 DataLoader 才将其锁页（需要 `PIN_MEMORY` 及具有锁页主机内存的加速器：
  CUDA 或 XPU；`kohakumoss_common.devices.streamed_type` 是加载器、SeqInfo 传输和
  卸载引擎共用的判断函数）。训练器传入 `kohakumoss.training.step.LayoutTransform`，
  在那里构建批次的 SeqInfo（文本长度为最终值时），并将其附加为 `hb.layout`，
  从而在训练线程之外构建布局。

<a id="worker-start-up-time"></a>
### 工作进程启动时间

DataLoader 逐个启动工作进程，且每次启动只有在新工作进程读取其大部分 pickle 参数后才返回；
工作进程先导入启动进程的主模块，再在导入参数所引用模块的同时读取参数
（torch、`kohakumoss_data`、变换的模块；训练器的 `LayoutTransform`
会连同所有后端一起引入 `kohakumoss_kernels`）。

| 启动方式 | 第 1 步前的等待 |
| --- | --- |
| `spawn` | 每个工作进程依次承担这些导入开销：如果启动器在顶层导入训练器，在 RTX 5090 主机（Linux）上使用 4 个工作进程，第 1 步会等待 46 s，与缓存模式无关 |
| `forkserver`（Linux） | 服务进程一次性预加载主模块、DataLoader 的工作循环，以及数据集和变换的模块，每个工作进程从它派生时都已导入这些模块，因此只需等待一轮导入 |

Windows 只有 `spawn`，其等待时间随 `NUM_WORKERS` 增长。

<a id="single-writer-files"></a>
## 单写入者文件

每个 KohakuVault 文件恰有一个写入进程；任意数量的进程可以读取它。

- 目录文件由持有其锁的进程写入（训练器的 rank 0；其他进程在它完成后读取）。
- 在线缓存为每个 rank 写一个分片文件。
- 离线缓存脚本和 `cache_merge.py` / `catalog_cache.py` 各自都是单进程。

<a id="tensor-store-files"></a>
### 张量存储文件

每个缓存文件都是一个只有一个写入者的 KVault（`TensorStore`）：
离线处理和 `cache_merge.py` 使用主文件，在线写入者使用 `<stem>.r<rank>.kv` 分片。

| 键 | 值 |
| --- | --- |
| `t:<key>` | 魔数 `KMT1`、数据类型代码 u8、ndim u8、填充 u16、缩放 f32、形状 u32 x ndim，然后是原始小端载荷 |
| `m:<key>` | 以 ASCII 数字表示的第一维（词元数），在 `t:<key>` 之后写入 |
| `idx:lengths` | u32 计数、已排序的 16 字节键、对应的 i32 长度 |

fp8 值为 e4m3，带有以 fp32 计算的逐张量 fp32 缩放 `absmax / 448`
（全零张量为 1）。合并从每个复制条目的 `t:` 头中获取长度，
因此写入者在两次写入之间停止的分片仍能合并。

MOSS 形状下的大小：1 MP 潜变量为 64 x 64 x 128 bf16 = 1 MiB；
文本嵌入为 L x 4096 bf16 = 每词元 8 KiB，因此 300 词元的描述文本每个变体约为 2.4 MiB。
