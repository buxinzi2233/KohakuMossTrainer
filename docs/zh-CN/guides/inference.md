[English](../../guides/inference.md)

<a id="inference"></a>
# 推理

`kohakumoss.inference.MossPipeline` 使用官方 MOSS 采样器生成图像：

- 在 shift-5 sigma 网格上执行 Euler，吸附到 1000 步训练网格；
- CFG 3，负向分支跳过第 22 块。

`scripts/sample/sample.py` 从命令行驱动它（[CLI](#cli)）；
`scripts/sample/compare_grid.py` 将检查点并排渲染
（[对比网格](#comparison-grid)）；Python API 接受相同的设置
（[Python](#python)）。

<a id="cli"></a>
## CLI

`scripts/sample/sample.py` 是一个 kogine 脚本：每项设置都是模块级配置项，
可在配置文件中或通过 `--set` 设置。`WEIGHTS` 是 README 中下载的 ComfyUI models 文件夹
（`ComfyUI/models`；`TRANSFORMER`、`TEXT_ENCODER`、`VAE` 指定其文件，默认是 int8 DiT 和
文本编码器），或 HF 流水线文件夹（`models/MOSS-Image-Base`，包含 `transformer/`、
`text_encoder/`、`vae/`）；见 [weights.md](weights.md)：

```bash
kogine run scripts/sample/sample.py --set WEIGHTS=ComfyUI/models \
    --set PROMPT="a cat" --set ADAPTERS=my_lora.safetensors:0.8 --set OUT=out/samples
```

配置文件包含相同的配置项：

```python
WEIGHTS = "ComfyUI/models"
PROMPTS = ["a cat", "a dog"]
SIZES = [(1024, 1024), (832, 1216)]  # (H, W) or "HxW" per prompt; the last repeats
SEEDS = [0, 1]  # missing seeds are SEED_BASE + index
ADAPTERS = [("my_lora.safetensors", 0.8)]
OUT = "out/samples"
```

`kogine run scripts/sample/sample.py --config my_sample.py --set STEPS=30`

无需手写配置，有两种方式可以获取：

- `kohakumoss-config new sample --weights <dir> --adapter <file>` 会为本地 GPU 写出一份：
  使用精度层及该显卡训练显存层的 `MEMORY_PROFILE`，文本编码器和 VAE 保持
  `auto`（[configs.md](configs.md)）。每张显卡都会得到一份。
- `scripts/examples/sample.ps1` / `sample.sh` 和 `grid.ps1` / `grid.sh` 会根据少量
  选项写出并检查一份配置，然后运行它
  （[scripts/examples/README.md](../../../scripts/examples/README.md)）。

<a id="knobs"></a>
### 配置项

- `PROMPT` 是单个提示词；`PROMPTS` 是列表。`--set PROMPTS=` 的值使用
  列表语法（`--set 'PROMPTS=["a, b", "c"]'`，或按逗号拆分的 `[a, b]`）；裸
  字符串视为一个提示词，其中的逗号会保留（compare_grid.py 的 `PROMPTS` 也相同）。
- `PROMPTS_FILE` 接受 `.txt`（每行一个提示词），或由字符串或
  `{"prompt", "negative", "size": [H, W], "seed"}` 对象构成的 `.json` 列表。
  项目的顺序为 `PROMPTS`、`PROMPT`，然后是文件。
- `PRECISION = "fp16"` 配合 `FP16_FIX = "online"` 和 `FP16_FIX_KWARGS = {"c": 0.0625}`
  可选择 fp16 修正；`PRECISION = "auto"` 在显卡支持 bf16 时使用 bf16，否则使用 fp16。
- `VARIANT` 是架构预设，与训练中一样，为 `"base"`（默认）或 `"sft"`；若根目录
  包含 `base/` / `sft/` 文件夹，则读取 `WEIGHTS/<VARIANT>/`，而流水线目录
  按原样读取（[权重根目录](#weights-root)）。
- 加载任何权重之前都会检查所有配置项（`kohakumoss.inference.knobs.validate`）；
  所有问题会在一个错误中一并报告。

<a id="shared-with-the-train-script"></a>
### 与训练脚本共享的设置

配置项的名称和值与训练脚本一致（`PRECISION`、`BASE_PRECISION`、
`MEMORY_PROFILE`、`TEXT_ENCODE`、`VAE_ENCODE`、`VARIANT`、...），因此可以
在两者之间复制值，但以下情况除外：

| 值 | 有效范围 |
| --- | --- |
| `MEMORY_PROFILE="manual"` 配合仅限训练的配置项 | 训练 |
| `MEMORY_PROFILE=""`（整个 DiT 常驻） | 采样 |
| `PRECISION="fp32"` | 采样 |
| `MEMORY_PROFILE="auto"` | 两者皆可；采样时，它会选择最快且单张 1024x1024 CFG 样本的 DiT 前向（推理估算）能够装入设备空闲内存的配置档，日志记录为 `MEMORY_PROFILE 'auto' -> <profile>`；文本编码器和 VAE 随后围绕它安排自身的放置 |

训练脚本的采样配置项与此脚本的对应关系：

| 训练脚本 | sample.py |
| --- | --- |
| `SAMPLE_PROMPTS` | `PROMPTS` |
| `SAMPLE_NEGATIVE` | `NEGATIVE` |
| `SAMPLE_SIZE` | `SIZES` |
| `SAMPLE_SEEDS` | `SEEDS` |
| `SAMPLE_SKIP_LAYERS` | `SKIP_LAYERS` |
| `SAMPLE_STEPS` | `STEPS` |
| `SAMPLE_CFG` | `CFG` |
| `SAMPLE_SHIFT` | `SHIFT` |

<a id="output"></a>
### 输出

- 每个提示词输出一张 PNG，从 `OUT` 中现有的最大 `NNNN_` 索引之后开始编号。
- 每次运行输出一个 `metadata-<first index>.json`，包含所有配置项和解析后的
  流水线设置；复用 `OUT` 会保留之前的图像。
- 运行会记录每个阶段及其耗时，并为每个批次显示进度条
  （[logging.md](logging.md)）。

<a id="python"></a>
## Python

```python
import torch
from kohakumoss.inference import MossPipeline

pipe = MossPipeline.from_pretrained(
    "ComfyUI/models",            # a ComfyUI models folder, a pipeline dir, or a root with base/ (and sft/)
    dtype=torch.bfloat16,
    adapters=[("my_lora.safetensors", 0.8)],
    adapter_mode="merge",        # or "bypass"
    text_plan={"memory": "auto"},  # TE: "auto" | "resident" | "offload" | "ring" | "cpu"
    vae_plan={"memory": "auto"},   # VAE: "auto" | "resident" | "offload" | "ring" | "cpu"
)
out = pipe.generate(
    ["a cat on a windowsill", "a lighthouse at dusk"],
    sizes=[(1024, 1024), (832, 1216)],
    seeds=[0, 1],
    steps=50, cfg=3.0, shift=5.0,
)
out.images[0].save("cat.png")
```

下述设置为 `from_pretrained` / `generate` 的参数；若
`scripts/sample/sample.py` 的配置项名称不同，会在括号中注明。

<a id="sampling"></a>
### 采样

- **打包 CFG**（`cfg_mode="packed"`，默认）：一个批次的有条件和无条件样本
  在一次打包前向中运行；`two_pass` 则以每次一半的 token 分两次前向运行。
- **逐样本确定性**：样本 i 的噪声和条件仅取决于其提示词、尺寸和种子：
  其他样本的内容不会影响它，它在批次中的位置也无关紧要（打包注意力和文本打包
  会隔离样本）。从逐位一致的角度看，样本还依赖批次的形状、其他样本的尺寸和
  token 数，这些因素决定了 GEMM、注意力和文本编码器的问题形状，进而决定
  舍入结果；相同的批次组成可以逐位复现一个样本。
- **传入嵌入**：`prompt_embeds` 会跳过提示词编码；除非提供 `negative_embeds`，
  否则仍会编码 `negative_prompts` 字符串。
- **线性代数**：拥有自身模型的 CUDA 流水线会在每次 `generate` 调用期间
  选择 cuSOLVER（OFT 的批量 Cayley 逆），调用结束后恢复先前的后端。

<a id="weights-root"></a>
### 权重根目录

`path` 是一个 ComfyUI models 文件夹：`transformer`、`text_encoder_file`、`vae_file`
（`TRANSFORMER`、`TEXT_ENCODER`、`VAE`）指定其文件，各自为空字符串 `""` 时使用默认值
（`moss_image_<variant>_int8_convrot`、`qwen3.5_9b_int8_convrot`、`flux2-vae_small`，
各自回退到下一个存在的文件；[weights.md](weights.md#comfyui-layout)）。

否则 `path` 是一个流水线文件夹，或包含 `base/`（以及 `sft/`）子文件夹的根目录。
`variant`（`VARIANT`，默认 `"base"` 或 `"sft"`）是架构预设：

- 每个组件在 `path/variant` 持有其权重（即含有 safetensors 文件的组件文件夹）时从该
  文件夹读取，否则从 `path/base` 读取（会记录日志）。只包含 transformer 权重
  （其 `text_encoder/` 和 `vae/` 只有配置而无权重）的 SFT 文件夹会使用 base 的
  文本编码器和 VAE。
- 没有 `path/variant` 文件夹时，`path` 本身就是流水线文件夹。
- 如果根目录没有所请求变体的文件夹（例如在只有 `base/` 的根目录上请求 `sft`），
  则报错。
- 使用的文件夹记录在 `out.settings["sources"]` 中。

<a id="dit-memory-plan"></a>
### DiT 内存计划

`memory=` 一个配置档（`MEMORY_PROFILE`、`MEMORY_OVERRIDES`；任意目录条目，
[profiles.md](../concepts/profiles.md)）会安排 DiT 块权重的放置：

| 配置档 | DiT 块 |
| --- | --- |
| `speed`（或不指定配置档） | 整个 DiT 常驻 |
| `balanced` | 通过环形缓冲区流式传输，16 块常驻 |
| `low` | 流式传输，4 块常驻 |
| `minimal` | 流式传输，2 块常驻 |
| `manual` | `STREAM` / `STREAM_RESIDENT` / `PINNED` |

- 推理会应用配置档的权重放置。无论使用哪个配置档，基础精度都有独立的配置项
  （`BASE_PRECISION`，[fp8 / MXFP8 计算](#fp8--mxfp8-compute)）。
- 适配器在计划完成 DiT 放置后附加。对于量化或流式传输的基础模型，合并操作
  会改为旁路运行并发出 WARNING（合并需要 16/32 位稠密基础模型），适配器
  日志行会注明实际模式（`merge + bypass (N layers on stored bases)`）。
- 每次调用都在 `inference_mode` 下运行，文本编码、采样和解码之后，
  分配器缓存都会归还给驱动。

分阶段内存（fp16、1024^2、TE ring fp8、RTX 3090；`scripts/bench/infer/memory.py`，
接受 `--memory` 和 `--precision`；
[sampling-memory.md](../internals/sampling-memory.md) 将峰值按阶段分解）：

| 配置档 | 常驻 DiT GiB | 每次调用峰值 GiB |
| --- | --- | --- |
| `speed` | 9.63 | 15.07 |
| `balanced` | 4.14 | 7.24 |

<a id="text-encoder-placement"></a>
### 文本编码器放置

`text_plan`（`TextEncoderPlan` 或其字段；`TEXT_ENCODE`，与训练使用相同的字典）。
`memory` 为 `auto`（默认）时，会选择能与 DiT 共存且保留 `reserve_gib` 空闲内存的
最快放置方式（默认预留量：单张 1024^2 CFG 样本的 DiT 前向工作集）：

| `memory` | 放置方式 |
| --- | --- |
| `resident` | bf16 下在设备上占用 14.8 GiB |
| `offload` | 主机副本，每次编码调用时移入，调用后释放 |
| `ring` | 每次通过小型环形缓冲区传输一层解码器；调用期间设备上约占 2.3 GB，调用之间不占用（`stream` 也被接受为相同的值） |

- `precision="int8"` / `"fp8"` 将权重和每次调用的流量减半；
  `fp8_compute`（配合 row / tensor / block 的 `fp8_recipe`）和 `mxfp8`
  在 sm89+ 上以 fp8 执行 GEMM 本身（常驻占用为 8.4 / 8.8 GiB，而非 14.8 GiB；见
  [fp8 / MXFP8 计算](#fp8--mxfp8-compute)）。
- `dtype` 为 `auto`（bf16，不支持 bf16 时为 fp16）、`bf16` 或 `fp16`。
- `free_after_use=True` 在编码器第一次执行后释放它（编码该调用的所有提示词）。
- 一次调用的提示词一起编码（`layout="packed"`，每次内核调用最多 `batch` 个
  提示词）；见 [text-encoder.md](../internals/text-encoder.md)。
- 用户指定的放置方式会按原样使用。一次调用的分阶段内存见
  [sampling-memory.md](../internals/sampling-memory.md)。
- **文本设备**：`text_device="cpu"`（`TEXT_DEVICE = "cpu"`）在 CPU 上运行文本
  编码器，仅将嵌入移到计算设备。此时 9B 编码器在 bf16 下需要约 ~18 GB
  主机内存，不占显存；编码速度取决于 CPU。训练仅在离线缓存阶段使用 CPU。

<a id="vae-placement"></a>
### VAE 放置

`vae_plan`（`VaePlan` 或其字段；`VAE_ENCODE`，与训练中相同）：

- `auto`（默认）从以下方式中选择第一个能与 DiT 共存的：未分块的 `resident`、
  未分块的 `offload`（停放在主机上，每次解码时移入）、同样两种方式的分块版本，
  最后是流式分块执行。
- `stream` 和 `cpu` 是另外两个值；`tile` 强制分块执行。
- `VAE_DTYPE` 设置其 dtype。

<a id="fp8--mxfp8-compute"></a>
### fp8 / MXFP8 计算

在 sm89+ 上，`BASE_PRECISION = "fp8_compute"`（配合 `FP8_RECIPE` `"row"` / `"tensor"` /
`"block"`）或 `"mxfp8"` 会在加载时一次性量化 DiT 的冻结线性层（e4m3 权重，
每个 1 byte，另加其缩放值），并以 fp8 执行其 GEMM，每次调用都量化激活。
`BASE_PRECISION` 的 int8 / fp8 是仅权重量化格式。

- `BASE_PRECISION = "auto"`（默认）按 DiT 文件存储的内容运行：ComfyUI int8 文件使用其
  int8 ConvRot 基础权重（无配置档时常驻），16 位文件使用 16 位；`"compute"` 会反量化
  int8 文件（[weights.md](weights.md#dit)）。
- 量化的 `BASE_PRECISION` 会让 DiT 使用 `speed` 配置档，除非 `MEMORY_PROFILE`
  明确指定了配置档（Python 中为 `memory_overrides={"base_precision": ..., "fp8_recipe":
  ...}`）。
- 文本编码器接受相同的格式（`TEXT_ENCODE["precision"]`，默认 `auto`：ComfyUI int8 /
  W4A8 文件保留其存储的层）。

RTX 5090、1024^2、2 个提示词配合 CFG 执行 20 步（`scripts/probe/fp8_inference.py`）：

| DiT 基础模型 | 权重 GiB | s / step | 峰值 GiB | 速度预测相对 fp64 的余弦相似度 |
| --- | --- | --- | --- | --- |
| bf16 | 9.66 | 0.875 | 10.49 | 0.99979 |
| fp8_compute row | 5.01 | 0.567 | 6.16 | 0.99815 |
| fp8_compute tensor | 5.01 | 0.570 | 6.15 | 0.99796 |
| fp8_compute block | 5.01 | 0.638 | 6.15 | 0.99817 |
| mxfp8 | 5.33 | 0.483 | 6.48 | 0.99798 |

参考结果是 bf16 权重在主机上的 fp64 前向。少量短提示词使用 fp8 编码会比
bf16 慢（在该规模下，每个 GEMM 都受启动开销限制）；格式和内核见
[quant-kernels.md](../internals/quant-kernels.md)。

<a id="fp16"></a>
### fp16

在不支持 bf16 的 GPU 上，传入 `dtype=torch.float16`（`PRECISION="fp16"`），并选择以下之一：

- `fp16_fix`（`"online"` 或 `"fp32_residual"`，配合 `fp16_fix_kwargs={"c": 0.0625}`），
  或
- 通过 `transformer=...`（`TRANSFORMER`）传入 fp16 修正检查点，其中自带折叠
  （`FP16_FIX` 保持 `"none"`）。

参见 [numerics.md](../concepts/numerics.md)。

<a id="kernel-tuning"></a>
### 内核调优

`tune`（`TUNE`；默认 `"off"`）。推理会绑定该显卡已存储的决策，否则绑定默认
配置；`"table"` 会在缺失决策首次使用时调优并存储它。

| `TUNE` | 效果 |
| --- | --- |
| `"off"`（默认） | 绑定调优表中为此显卡存储的决策（由训练运行或 `kohakumoss-tune` 填充），否则使用默认配置；不做任何计时。LyCORIS 以相同方式绑定其已存储的 tile 计划 |
| `"table"` | 还会在缺失决策首次调用时进行调优（该次调用更慢）并存储，且在每次 generate 之后测量排队等待的内核与 compile 对比问题 |

参见 [tuning.md](../internals/tuning.md#pipelines)。

<a id="training-time-samples"></a>
### 训练期间采样

`MossPipeline.from_components(backbone, text_encoder, vae)` 原地共享训练模型的
模块（在 `no_grad` 下）；采样提示词在训练开始时编码一次，并作为嵌入传入。

<a id="step-diagnostics"></a>
## 步骤诊断

采样器的每一步都会以 INFO 记录耗时；对于流式传输的 DiT，还会记录：

- 权重环形缓冲区复制的字节数；
- 复制流忙碌的秒数，以及计算流等待复制的秒数；
- 主机同步次数。

在加速器上，每步会记录一个计时事件，设备执行当前步骤时，主机会将下一步入队。
批次的事件在结束时统一读取一次，随后记录其步骤日志行；步骤耗时是相邻
步骤事件之间的设备时间。

每个批次结束时都会给出摘要：第一步（JIT 编译和首次复制发生于此）以及
其余步骤的均值，还有复制重叠率（1 - wait / copy）。相同数据也会出现在
`settings["step_stats"]` 和采样脚本的 `metadata-<first index>.json` 中。

解读方式：

- 步骤慢但等待很少，说明受计算限制（未调优或 eager 算子）；
- 等待很长，说明复制没有重叠。

冻结权重环会将前 `stream_resident - 2` 个块保留在设备上，并通过两个槽位循环
传输其余块，因此 `balanced`（`stream_resident` 16、42 块）会在设备上保留
14 块，每步复制 28 块（6.34 GB，每块 0.21 GiB）。
RTX 3090（Linux、fp16 修正、1024^2、CFG packed、TUNE=table），首步之后的 s/step：

| DiT 放置方式 | s/step | 每步复制量 | 重叠 |
| --- | --- | --- | --- |
| 常驻 | 1.557（首步 6.92） | - | - |
| `balanced` | 1.448 | 0.49 s 内复制 6.34 GB | 无等待 |
| `minimal` 配合 `BASE_PRECISION="int8"`（int8 块通过两个槽位传输） | 1.568 | 4.76 GB | 98% |

<a id="comparison-grid"></a>
## 对比网格

`scripts/sample/compare_grid.py` 将适配器检查点与基础模型进行对比：

```bash
kogine run scripts/sample/compare_grid.py --config my_grid.py
```

```python
WEIGHTS = "ComfyUI/models"
CHECKPOINTS = [
    "runs/quality/quality-lora8/checkpoints/step_00000100/trainable.safetensors",
    "runs/quality/quality-lora8/checkpoints/step_00000200/trainable.safetensors",
    "runs/quality/quality-lora8/checkpoints/step_00000400/trainable.safetensors",
]
PROMPTS = ["a red brick house beside a lake", "a lighthouse at dusk"]
SEEDS = [0, 1]
OUT = "runs/quality/quality-lora8/grid"
```

渲染内容：

- 权重依次为基础模型，然后是 `CHECKPOINTS` 中的每一项（路径或
  `(path, multiplier)`；裸路径使用 `MULTIPLIER`），每组权重都针对 `PROMPTS` 渲染。
- 对不同权重，每个提示词都保持相同的种子（`SEEDS`，否则为 `SEED_BASE` + 索引）
  和采样器设置，并使用对应的 `SIZES` 条目（逐提示词，最后一项重复使用，与
  sample.py 相同）。
- 设备、精度和文本编码器配置项与 `sample.py` 相同。

运行方式：

- DiT 只加载一次，所有列的提示词也只编码一次（因此
  `TEXT_ENCODE = {"free_after_use": True}` 会在该次编码后释放文本编码器）。
- 每个检查点在轮到它时加载到 DiT 上，完成后移除
  （`ADAPTER_MODE` 为 bypass 或 merge，两者都会精确撤销），因此基础模型就是未经修改的模型。
- 训练检查点在其训练时的折叠空间中读取
  （[adapters.md](../concepts/adapters.md#inference)），不受 DiT 文件所带折叠的影响。

布局和标签：

| 配置项 | 效果 |
| --- | --- |
| `GRID_LAYOUT` | `"prompt-cols"`（默认）：提示词为列，权重为行；`"prompt-rows"` 将两者交换 |
| `LABELS` | 权重标签；默认从训练检查点路径提取 `step N`。提示词标签为 `seed: prompt` |
| `LABEL_LINES` | 标签在单元格内按词换行，最多使用此数量的行 |
| `FONT_SIZE` | 标签字体的像素大小（0 = 图像宽度 / 32，最小 14） |

`OUT` 中会生成 `grid.png`、每张图像 `r<prompt>_c<weight>.png`，以及 `metadata.json`
（列、行、文件、配置项）。
