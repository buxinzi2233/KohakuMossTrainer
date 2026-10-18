[English](../../guides/configs.md)

<a id="configs-and-presets"></a>
# 配置与预设

每个脚本都是 kogine 脚本：设置采用模块级 UPPER_CASE 配置项，配置文件则是包含同名变量的 Python 文件。

- `kogine run <script> --config <file> --set KEY=VALUE` 加载文件，应用每个 `--set`（强制转换为配置项的类型），然后调用脚本的 `main()`。
- 配置通过 `use_config("<file>")` 组合：先应用指定文件的配置项，后面的调用覆盖前面的调用，文件自身的赋值覆盖上述所有设置。
- `use_config` 会替换整个值，字典也不例外。

最快的入门方式是生成针对你的 GPU 和数据集的起始配置（[通过标志生成起始配置](#starter-configs-from-flags)）；也可以扩展固定[配方](#recipes)或任意训练配置（[从基础配置开始](#from-a-base-config)）。

<a id="starter-configs-from-flags"></a>
## 通过标志生成起始配置

```bash
kohakumoss-config new train --weights path/to/MOSS-Image-Base --data path/to/dataset
kohakumoss-config new sample --weights path/to/MOSS-Image-Base --adapter runs/moss/my-lora/export/my-lora-comfy.safetensors
kohakumoss-config list
kohakumoss-config check configs/user/my-lora.py
```

`new train` 会在克隆仓库的 `configs/user/` 中（已被 git 忽略）写入两个文件；通过 PyPI 安装时则写入当前目录（可用 `--out` 指定其他位置）：

| 文件 | 内容 |
| --- | --- |
| `<name>.py` | 训练配置：每个预设层各有一行 `use_config`，随后像 `new --from` 一样列出所有配置项，其中 `WEIGHTS`、`DATASETS`（数据层中填入你的路径的条目）、`RUN_NAME` 以及通过 `--steps` / `--prompt` / `--gpus` 指定的设置处于启用状态 |
| `<name>_data.py` | 数据准备配置：`use_config("<name>.py")` 加上缓存编码器（针对这些权重的 `kohakumoss.encoders.text_encoder`、`kohakumoss.encoders.vae_encoder`） |

数据脚本读取与训练相同的数据配置项，因此缓存键天然一致；编码器配置项不会写入训练配置，因为训练配置的配置项检查会拒绝训练脚本未声明的名称。两个文件的文档字符串都会按顺序列出运行命令（检查、文本缓存、潜变量缓存、训练）。

未指定的标志从本机 GPU 读取：

- `--vram`：设备 0 的内存容量，以精确的 GiB 计。显卡会选择在 `--vram-tolerance` 容差内能容纳的最大显存层；该容差是层大小的一定比例（默认 0.05）：24 GB 显卡报告约 23.99 GiB，会得到 `vram_24g`；使用 `--vram-tolerance 0` 则会得到小一档的层（[选择显存层](#choosing-a-vram-layer)）。
- `--precision`：架构支持 bf16 时使用 bf16，否则使用 fp16。
- `--gpus N`（Linux）添加并行层（[多 GPU 层](#multi-gpu-layers)）。

在写入任何内容之前，组合后的配置项会经过训练脚本的配置项检查；无效组合会被拒绝，并列出所有问题，而尚不存在的路径只会产生警告。

`new sample` 为 `scripts/sample/sample.py` 写入 `<name>.py`（默认名称为 `sample`）；它按相同的显存规则选择 DiT 的放置方式（[采样层](#sample-layers)）。

<a id="from-a-base-config"></a>
## 从基础配置开始

```bash
kohakumoss-config new --from lora_24g_bf16 my_run.py
kohakumoss-config wizard
```

`new --from <base> <file>` 写入一份通过 `use_config("<base>")` 扩展基础配置的配置文件（基础配置可以是 `kohakumoss-config list` 中的配方名称，也可以是任意训练配置路径），因此后续对基础配置的更改会传递过来。在这行下方，文件按 `scripts/train/moss.py` 的分节和顺序列出它的每个配置项，每项都附有脚本中的单行注释：

- `WEIGHTS`、`DATASETS`、`RUN_NAME` 和 `STEPS` 以基础配置的值启用；
- 其他所有配置项均以基础配置给出的值列出，但处于注释状态。

编辑 `WEIGHTS` / `DATASETS`，取消你要更改的配置项的注释，然后运行 `kogine run scripts/train/moss.py --config my_run.py`。

`wizard` 会询问 GPU 内存（默认：设备 0 的容量）、精度（默认：架构支持时使用 bf16）、算法、数据集布局与路径、权重路径、步数、GPU 数量、运行名称和文件，然后根据下面的预设层写入同样带注释的文件，以及数据准备配置。

<a id="layers"></a>
## 层

一个层设置一个方面，并组合在基础配置之后。训练层位于 `configs/moss/<group>/`，采样层位于 `configs/sample/<group>/`。训练层按组合顺序如下：

| 组 | 层 | 设置内容 |
| --- | --- | --- |
| `_base` | `base_lora`（`base_full` = base_lora + algo_full） | 权重 / 数据占位项、目标函数、调度、缓存、采样 |
| `algo` | `lora`、`loha`、`lokr`、`oft`、`dora`、`full` | ALGO、DIM、ALPHA、ALGO_KWARGS、DROPOUT、LR；`full`：优化器、EMA、导出、`ckpt-resident` |
| `data` | `kohya`、`folder`、`folder_nl`、`jsonl`、`parquet`、`webdataset`、`moss_shard` | 包含每个数据源参数的 DATASETS 条目模板、CAPTION、ASPECT_RANGE |
| `vram` | `6g`、`8g`、`12g`、`16g`、`24g`、`auto`、`24g_full` | MEMORY_PROFILE、MAX_TOKENS、TEXT_ENCODE、VAE_ENCODE；`6g`：离线缓存；`6g` / `8g`：SAVE_MINUTES 30；`24g_full`：AdamW、关闭 EMA |
| `precision` | `bf16`、`fp16` | PRECISION、FP16_FIX（fp16：`online`，c = 1/16） |
| `parallel` | `ddp`、`fsdp`、`pipeline`、`threaded` | PARALLEL、DEVICES 以及引擎所需的设置（见 [multi-gpu.md](multi-gpu.md)） |

顺序在两个地方很重要：

- 数据层在显存层之前，因此由显存层决定 MAX_TOKENS；
- 显存层在算法层之后，因为 `24g_full` 会替换优化器（可训练权重流式传输运行 AdamW）。

<a id="algorithm-arguments"></a>
### 算法参数

`ALGO_KWARGS` 以 LyCORIS 自己的参数名称原样传给 LyCORIS 模块构造函数；配置项检查会拒绝模块不接受的名称。算法层按 LyCORIS 默认值明确列出主要参数（层本身修改的值除外），因此可以在配置中就地编辑：

| 算法 | ALGO_KWARGS | |
| --- | --- | --- |
| LoRA / LoCon（`lora`）、LoHa（`loha`） | `use_tucker`、`use_scalar`、`rank_dropout_scale` | 卷积使用 Tucker、可训练标量、rank dropout 后重新缩放 |
| LoKr（`lokr`） | `factor`（-1 = 接近正方形的拆分；LyCORIS 默认值）、`full_matrix`、`decompose_both`、`use_tucker`、`rank_dropout_scale` | `full_matrix=True` 在任意 DIM 下都保持 w1 / w2 为完整矩阵；DIM >= 较大因子的一半时也如此 |
| OFT（`oft`） | `constraint`、`rescaled` | 旋转的范数约束、可训练的重新缩放 |

- Dropout 有独立的配置项 `DROPOUT = {"input": ..., "rank": ..., "module": ...}`，映射到 LyCORIS 的 `dropout`、`rank_dropout` 和 `module_dropout`（ALGO_KWARGS 不接受这些名称）。
- DoRA 使用 `DORA = True`（LyCORIS 的 `weight_decompose`）。

<a id="choosing-a-vram-layer"></a>
### 选择显存层

每个适配器层都是一个目录配置档：

| 层 | MEMORY_PROFILE | MAX_TOKENS | 文本编码器 / VAE |
| --- | --- | --- | --- |
| `24g` | `balanced` | 12288 | TE 环形缓冲，每轮 8 个批次 |
| `16g` | `low` | 8192 | TE 环形缓冲，每轮 4 个批次 |
| `12g` | `ckpt-ring4-actoff-optstream-gradoff` | 6144 | TE 环形缓冲；VAE 卸载 |
| `8g` | `minimal` | 4352 | TE 环形缓冲，每轮 8 个批次；VAE 卸载、分块 |
| `6g` | `ckpt-ring2-actoff-optcpu-gradoff` | 4352 | 离线潜变量和文本缓存（采样提示词使用基础配置的 TE 方案） |

起始配置的选择如下：

| 显卡 | 层 |
| --- | --- |
| 适配器，最高 24 GiB | 在 `--vram-tolerance` 容差内显卡能容纳的最大层（显卡至少具有 `size x (1 - tolerance)` GiB；默认 0.05 时，`24g` 需要 22.8 GiB） |
| 适配器，超过 24 GiB | `auto`（规划器选择估计内存能容纳的最快配置档，见[配置档](../concepts/profiles.md)） |
| 适配器，低于最小层（6 GiB） | 拒绝；消息中会指出 `--vram` 和 `--vram-tolerance` |
| 全量微调 | `24g_full`，超过 24 GiB 时为 `auto`；无法在容差内容纳 24 GiB 的显卡会被拒绝 |

- 精度来自精度层；量化基础模型通过 `BASE_PRECISION` 配置项设置（12 GB 和 XPU 配方设置为 `int8`）。
- 8 GB 和 6 GB 目标由基准矩阵的 `cap` 运行检查（将分配器上限设在目标以下，nvidia-smi 峰值不超过目标即通过）。各显卡上各层的单步耗时和 nvidia-smi 峰值见基准矩阵的速度表（`report.md`，[bench-matrix.md](../performance/bench-matrix.md#report)）。

<a id="the-text-encoder-in-the-vram-layers"></a>
### 显存层中的文本编码器

`8g` 及以上显存层会设置文本编码器方案 `TEXT_ENCODE`（[text-encoder.md](../internals/text-encoder.md)）；`6g` 保留基础配置的方案。

- 只要启用了训练期间采样或在线文本缓存，训练器就会加载文本编码器。
- 在最高 24 GiB 的显卡上，编码器的 15.63 GiB（bf16）无法与 9.47 GiB 的 transformer 一起容纳，因此这些层通过层环形缓冲流式传输编码器（`memory="ring"`，一轮期间约 2 GiB，轮次之间不占用显存），每轮编码 `ahead` 个批次。
- 使用离线文本缓存时，编码器为采样提示词运行一次后释放。
- `auto` 选择能与 DiT 步骤共存的最快放置方式。
- CPU 仅用于离线预计算轮次（`scripts/data/cache_text.py`）。

<a id="host-ram"></a>
### 主机 RAM

根据组件大小，单 GPU 运行在主机内存中保留以下内容（bf16 基础模型；int8 / fp8 基础模型使 DiT 部分减半）：

| 部分 | 何时 | GiB |
| --- | --- | --- |
| 训练器进程及其 `NUM_WORKERS` 个加载器工作进程（Windows 上为生成的进程，每个都导入 torch） | 始终 | 基准矩阵为每次运行预留 8（`kohakumoss_bench/plan/hostmem.py`） |
| DiT，在放置之前加载到主机 | 构建时 | 9.47 |
| 文本编码器，读入主机 RAM | 构建时，当 `SAMPLE_EVERY > 0` 或 `TEXT_CACHE` 不是 `offline` 时；使用离线文本缓存时，在处理完采样提示词后释放 | 15.63 |
| 流式传输的 DiT 块，使用锁页内存（共 42 个块，每块 0.21；设备保留 `stream_resident` 个块） | 流式传输配置档 | 约 5.5（`balanced`）至 8.4（两个常驻块） |
| 卸载的激活值（块输入，每块一份，使用向上取整到 2 的幂的锁页缓冲区） | `act_offload` | 4352 个 token 时约 1，8192 时约 2 |
| 适配器主副本、矩估计和梯度累加和（每个可训练参数 16 字节） | `optim_state` 为 `stream` / `cpu`，`grad_offload` | `attn-mlp` 上 LoRA DIM 32（74 M 参数）约 1.1 |

各层合计：

| 层 | 构建峰值 | 训练步骤期间 |
| --- | --- | --- |
| `8g` | 接近 32 GiB（进程、流式传输块和文本编码器） | 约 19 GiB |
| `24g`（`balanced`） | 接近 29 GiB | 约 14 GiB |

设置 `SAMPLE_EVERY = 0` 且两个缓存均为离线时，只读取缓存，文本编码器留在磁盘上，因此去掉了 15.63 GiB 的构建峰值。全量微调见 [full-finetune.md](full-finetune.md#host-ram)。

<a id="multi-gpu-layers"></a>
### 多 GPU 层

`kohakumoss-config new --gpus N` 选择并行层：

| 运行 | 层 |
| --- | --- |
| 适配器 | `ddp` |
| 全量微调 | 当每个 rank 的 FSDP2 占用（`fsdp` 层的配置档，所有内容分片到 N 个 rank，按内存估计）在 `--vram-tolerance` 容差内能放入显卡时使用 `fsdp`，否则在全量微调显存层的换入换出配置档上使用 `ddp`（[full-finetune.md](full-finetune.md)） |
| `--vram auto` | `ddp` |
| `pipeline`、`threaded` | 手动组合 |

- `ddp` 保持显存层的配置档不变（每个配置档都可在 DDP 下运行）。
- `fsdp`、`pipeline` 和 `threaded` 将配置档替换为 `ckpt-resident`（常驻的计算精度权重、GPU 状态），并关闭采样。
- 由于 `use_config` 会替换整个字典，`kohakumoss-config new` 会按层顺序合并所有设置 MEMORY_OVERRIDES 的层，并将合并后的 MEMORY_OVERRIDES 写入起始配置；手写配置若叠加两个这样的层，需要自行合并字典。

<a id="sample-layers"></a>
### 采样层

| 组 | 层 | 设置内容 |
| --- | --- | --- |
| `_base` | `base_sample` | 官方采样器（50 步、CFG 3、shift 5）、占位项 |
| `precision` | `bf16`、`fp16` | PRECISION、FP16_FIX（fp16：`online`，c = 1/16） |

采样配置项沿用训练脚本的名称和值（`PRECISION`、`BASE_PRECISION`、`MEMORY_PROFILE`、`TEXT_ENCODE`、`VAE_ENCODE`、`VARIANT`）。

`new sample` 允许使用任何显卡：

- 它写入该显卡训练显存层的 `MEMORY_PROFILE`（低于 6 GiB 时使用 `6g` 层的配置档；高于最大层时不设配置档，整个 DiT 常驻），因此采样时保留 DiT 的方式与该显卡上的训练相同；
- 它不设置 `TEXT_ENCODE` / `VAE_ENCODE`：两者均为 `auto`，即能与 DiT 共存的最快放置方式（[sampling-memory.md](../internals/sampling-memory.md)）。

基础权重大小为 9.47 GiB（transformer，bf16）、15.63 GiB（文本编码器，bf16）和 0.23 GiB（VAE）。

<a id="recipes"></a>
## 配方

`configs/moss/recipes/` 是上述层的固定组合（加上 RUN_NAME 和配方特定的值）。可以直接运行、在自己的文件中通过 `use_config` 引用，或用 `kohakumoss-config new --from <name>` 扩展：

| 配方 | 组合 |
| --- | --- |
| `lora_24g_bf16` | base_lora + vram_24g + precision_bf16 |
| `lora_16g_fp16` | base_lora + vram_16g + precision_fp16（V100、2080 Ti） |
| `lora_12g_stream` | base_lora + vram_12g，`BASE_PRECISION="int8"` |
| `lora_xpu_b50` | base_lora + vram_16g + precision_bf16，`BASE_PRECISION="int8"`，6144 个 token（Intel Arc Pro B50） |
| `lora_ddp` | base_lora + vram_24g + precision_bf16 + parallel_ddp（两个 GPU，Linux） |
| `dora` | base_lora + algo_dora + vram_24g + precision_bf16 |
| `lycoris_loha`、`lycoris_lokr`、`lycoris_oft` | base_lora + algo_loha / algo_lokr / algo_oft + vram_24g + precision_bf16 |
| `full_24g_offload` | base_full + vram_24g_full（一个 24 GB GPU） |
| `full_fsdp_4xv100` | base_full + precision_fp16 + parallel_fsdp，四个 GPU，8192 个 token |

`configs/` 下的其他组：

| 组 | 内容 |
| --- | --- |
| `configs/moss/quality/` | 质量测试配置（[见下文](#quality-test-configs)） |
| `configs/moss/scale/` | 多 GPU 扩展性层（[multi-gpu.md](multi-gpu.md)） |
| `configs/data/` | 用于 `scripts/data/` 各轮处理的数据准备配置（[datasets.md](datasets.md)） |
| `configs/sample/` | 采样层（[采样层](#sample-layers)） |

<a id="quality-test-configs"></a>
## 质量测试配置

`configs/moss/quality/` 中有两个配置：

| 配置 | 适配器 |
| --- | --- |
| `quality_lora8.py` | LoRA DIM 8、ALPHA 8、LR 5e-4 |
| `quality_lokr16.py` | LoKr factor 16、完整矩阵、DIM 10000、ALPHA 1、LR 2e-3 |

两者均训练 400 步，使用 AdamW，在 20 步预热后保持固定 LR，seed 0，每 100 步保存一个检查点（全部保留），使用流式传输文本编码器的在线文本缓存和离线潜变量缓存，采用 `auto` 显存层。

它们使用 bf16。使用 fp16 时，选择以下一种方式：

- 在线修复：`--set PRECISION=fp16 --set FP16_FIX=online --set
  FP16_FIX_KWARGS={"c":0.0625}`；
- fp16-fix 检查点：`--set PRECISION=fp16 --set WEIGHTS=<the fp16fix
  .safetensors> --set TEXT_ENCODER=<base>/text_encoder --set VAE=<base>/vae`，并将 `FP16_FIX` 保持为 `"none"`。训练器从文件的 `kohakumoss_fp16_fix` 元数据读取折叠信息，将模型标记为已折叠，并在导出适配器时撤销折叠（这样就能加载到未经修复的基础模型上）。单文件 `WEIGHTS` 只包含 transformer，因此每当运行需要加载 VAE / 文本编码器时，配置项检查都会要求提供 `VAE` / `TEXT_ENCODER`。

`scripts/sample/compare_grid.py` 渲染结果：行 = 提示词，列 = 基础模型以及随后各个检查点（[inference.md](inference.md)）。

<a id="validation"></a>
## 验证

`kohakumoss-config check <file>...` 可对任意配置运行随附配置检查。

`configs/` 下的每个配置都在文档字符串中注明其驱动的脚本（`scripts/<...>.py`，第一个生效），`tests/configs/test_shipped_configs.py` 会通过相应脚本的配置项检查来检查每份配置：

| 脚本 | 检查 |
| --- | --- |
| 训练 | `kohakumoss.training.knobs.validate` |
| 采样 | `kohakumoss.inference.knobs.validate` |
| 数据 | `kohakumoss_data.data_problems` |

层在其脚本的基础配置（`base_lora` / `base_sample`）之上接受检查，或在其通过私有 `_BASE = "<path>"` 指定的基础配置之上接受检查（kogine 忽略以 `_` 开头的名称）。
