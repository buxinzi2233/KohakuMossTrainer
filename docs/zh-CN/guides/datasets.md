[English](../../guides/datasets.md)

<a id="preparing-a-dataset"></a>
# 准备数据集

共四步：在数据配置中指向数据集、检查规划、缓存文本嵌入和潜变量，以及（可选）导入分片。以下命令使用数据配置 `configs/data/my_data.py`；通过 `kohakumoss-config new train` 生成的起始配置会写入 `configs/user/<name>_data.py`，可用它替代。

<a id="1-point-a-config-at-it"></a>
## 1. 在配置中指向数据集

`kohakumoss-config new train --data <dataset>` 会在训练配置旁写入数据准备配置 `<name>_data.py`，使用相同的数据配置项，并填好缓存编码器（[configs.md](configs.md)）。

手动操作时：让数据配置读取训练配置的数据配置项，并加上两行编码器设置（第 3 节），就能缓存与该运行读取的内容完全一致的数据：

```python
"""My data config. Drives scripts/data/inspect_data.py."""
from kohakuengine import use_config

use_config("../user/my-lora.py")   # the train config
TEXT_ENCODER_KWARGS = {"path": "models/MOSS-Image-Base", "device": "cuda"}
IMAGE_ENCODER_KWARGS = {"path": "models/MOSS-Image-Base", "device": "cuda"}
```

`configs/data/kohya/kohya_style.py` 是一个独立示例（设置 `data_root` 和 `WEIGHTS`）；与训练配置一样，它的缓存写入 `<data_root>/cache`，当运行的关键配置项（[data.md](../concepts/data.md#cache-keys)）与示例相等时，就会读取这些缓存。

<a id="kohya-style-folders"></a>
### Kohya 风格文件夹

```
dataset/
  5_some_artist/      image files + same-stem .txt tag files
  1_another/
```

只有条目中设置了 `"include_root": True`，才会读取直接位于 `dataset/` 下的图像（重复次数为 1）。kohya 扫描找不到图像时，会报错并指出布局，因此扁平图像文件夹需要 `include_root` 或 `--source folder`。

<a id="other-layouts"></a>
### 其他布局

每种布局都有一个数据层（`kohakumoss-config new train --source <name>`、`configs/moss/data/data_<name>.py`，明确列出每个数据源参数），每个数据源对应一个 `DATASETS` 条目：

```python
{"source": "folder", "path": "dataset", "txt_as": "tags"}            # any tree, .txt / .caption / .json sidecars
{"source": "jsonl", "path": "dataset/train.jsonl", "root": None}    # {"image": "a/1.png", "caption": "...", "tags": [...]} per line
{"source": "parquet", "path": "dataset/train.parquet", "image_col": "image", "bytes_col": None, "root": None}
{"source": "webdataset", "path": "shards/*.tar", "txt_as": "tags"}  # <key>.png + <key>.txt / <key>.json members
{"source": "moss_shard", "path": "dataset_shards"}                  # an imported directory (section 4)
```

清单数据源（`jsonl`、`parquet`）读取以下内容：

| 字段 | 键 |
| --- | --- |
| 图像路径 | `image` / `file` / `path` / `file_name`（`parquet`：`image_col` 列，或 `bytes_col` 中的编码图像），相对于 `root`（默认：清单所在的文件夹） |
| 描述文本 | `caption` / `text` / `caption_nl`（自然语言）或 `tags`（列表或逗号分隔字符串） |
| 元数据 | `artist`、`rating`、`weight` 和 `group` |
| 其他内容 | 保留在该行的 json 中，供 `json` 描述文本模板使用（[data.md](../concepts/data.md#sources)） |

`DATASETS` 中的多个条目按各自的 `weight` 混合。

<a id="2-check-what-the-planner-will-do"></a>
## 2. 检查规划器会如何处理

```bash
kogine run scripts/data/inspect_data.py --config configs/data/my_data.py
```

它会输出：

- 目录：行数、epoch 大小、尺寸和宽高比分位数、含 alpha 的占比；
- epoch 0 的规划：每批次样本数和 token 数、预算填充率、裁剪损失、放大占比、常见尺寸、超长样本的适配和丢弃情况；
- 描述文本示例。

在缓存之前，先在这里调整 `RESOLUTION_MODE`、`BATCHING`、`OVERLONG`、`TARGET_AREA`、`MAX_TOKENS`、`ASPECT_RANGE` 和 `CAPTION`。

<a id="3-cache-offline-mode"></a>
## 3. 缓存（离线模式）

```bash
kogine run scripts/data/catalog_build.py --config configs/data/my_data.py
kogine run scripts/data/cache_text.py --config configs/data/my_data.py
kogine run scripts/data/cache_latents.py --config configs/data/my_data.py
```

- 先运行文本处理：缓存文本长度后，`TOKEN_BUDGET="render"` 和 `"exact"` 会使用编码器的真实长度规划 token 批次和超长样本适配（默认值 `"estimate"` 使用目录中的字符数，不读取缓存）。
- 两轮处理都会跳过已经存在的键，因此可以在中断后继续。
- 在 Intel Arc GPU 上，将两个编码器 kwargs 中的设备都设置为 `"device": "xpu"`。

<a id="the-encoders"></a>
### 编码器

编码器来自数据配置：即第 1 节中的两行 `*_ENCODER_KWARGS`（空 `path` 会报错并指出该配置项）。`TEXT_ENCODER` / `IMAGE_ENCODER` 指定编码器工厂（点分路径；默认值是真实模型）：

| 工厂 | 加载内容 | `path` |
| --- | --- | --- |
| `kohakumoss.encoders.vae_encoder(path, device, dtype, ...)` | FLUX.2 VAE；返回其 `encode_images`，并附带 VAE 的计算规格 | `vae/` 文件夹或流水线根目录 |
| `kohakumoss.encoders.text_encoder(path, device, dtype, plan)` | Qwen3.5 文本编码器（`plan`：TextEncoderPlan 字段，在加载权重前针对设备进行检查） | 流水线根目录或 `text_encoder/` 文件夹 |

放置方式：

- 文本编码器通过 `TEXT_ENCODER_KWARGS["plan"]` 放置：不设置 `memory` 即为 `auto`，选择显卡能容纳的最快放置方式（`resident`，否则为 `ring`）；`{"memory": "ring"}` 或 `{"memory": "cpu"}`（在主机上编码）按指定值使用。
- 潜变量处理通过 `VAE_ENCODE` 以同样方式放置 VAE（未设置时为 `auto`）。
- 两个脚本每轮编码 `ENCODE_BATCH` 个条目：`cache_text.py` 中为描述文本（默认 512），`cache_latents.py` 中为图像（默认 8）。

编码器的计算方式来自数据配置通过 `use_config` 引用的训练配置：潜变量处理使用 `VAE_DTYPE` 和 `VAE_ENCODE`，文本处理使用 `TEXT_ENCODE` 的 `dtype`、`precision`、`quant_group` 和 `fp8_recipe`；kwargs 中设置的字段优先。编码器的 id 必须等于 `TEXT_ENCODER_ID`。

<a id="cache-keys"></a>
### 缓存键

两个编码器的规格都会进入缓存键（[data.md](../concepts/data.md#cache-keys)），因此缓存保存的内容与运行所依据的键一致，运行只有在计算方式相同时才会读取相应条目。

- 对目标尺寸、裁剪、翻转、描述文本模板、变体、种子或编码器规格的任何更改都会改变键：修改这些配置项后，重新运行处理。离线缓存未命中错误会指出存在差异的关键配置项，并输出重建命令。
- 潜变量键包含每张图像的内容标识；对于未导入的数据源，这是由文件大小、mtime 和绝对路径组成的文件指纹。因此，在数据集的一个挂载路径（Windows 上的 `D:/data/my_set`）构建的缓存，在另一个挂载路径（Linux 上的 `/mnt/data/my_set`）访问相同文件时会未命中：需要为每个挂载路径构建一次缓存，或者将数据集导入分片（第 4 节），分片使用每张图像内容的哈希作为内容标识。

<a id="online-caches"></a>
### 在线缓存

如果改为在训练配置中将 `LATENT_CACHE` / `TEXT_CACHE` 设置为 `"online"`，训练器会在训练期间编码缓存未命中的条目，并回写到相同的缓存文件；之后可通过 `scripts/data/cache_merge.py` 将在线缓存的各 rank 分片合并到主文件中（[data.md](../concepts/data.md#caches)）。

<a id="4-optional-import-into-shards"></a>
## 4. 可选：导入分片

在数据集条目中添加 `"import_to": "/path/out"` 并运行 `catalog_build.py`；然后使用 `{"source": "moss_shard", "path": "/path/out"}`。分片是大型顺序文件，相比大量小图像文件更适合网络存储。

对于以不同于当前布局的布局写入的目录文件：

- 扫描型数据源的缓存目录会通过重新扫描重建（产生一条指出文件的 WARNING）；
- 对于 `catalog.db` 是唯一行存储的 `moss_shard` 目录，这会构成错误：使用 `catalog_build.py` 和 `import_to` 重新导入其数据源。

每个扫描型数据源的目录都会缓存在磁盘上，并在每次启动时增量更新；`scripts/data/catalog_cache.py` 可显示、删除或压缩这些文件（[data.md](../concepts/data.md#the-catalog-cache)）。

<a id="throughput"></a>
## 吞吐量

```bash
kogine run scripts/bench/data/loader.py --config configs/data/my_data.py
```

写入 `runs/bench/data/loader_<mode>.json`（各工作进程数量下的批次/s、图像/s、联合 token/s）。
