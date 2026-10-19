[English](../../concepts/profiles.md)

<a id="memory-profiles"></a>
# 内存配置档

内存放置采用两种互斥模式之一，由 `MEMORY_PROFILE` 选择
（`scripts/train/moss.py` 和 `scripts/sample/sample.py` 使用相同配置项）：

- **具名配置档**（`balanced`、`ckpt-ring4-actoff`、……）设置所有内存配置项。
  在它旁边再设置内存配置项会触发验证错误，指出配置档和配置项名称。
- **`manual`** 从配置中获取所有内存配置项；保留为 `None` 的配置项使用下表中的默认值。

无论哪种模式，计划都在构建时一次性解析为具体的 `MemoryPlan`，
配置项验证会针对本地设备解析它，因此所有无效组合都将在加载任何权重之前，
通过一次启动错误报告。`MEMORY_OVERRIDES`（包含原始计划字段的字典）
叠加应用于任一模式之上。

<a id="choosing-a-profile"></a>
## 选择配置档

四个简名覆盖常见范围，按速度从快到慢排列。每个都是[配置档目录](#the-catalog)中的条目；
使用简名的运行会记录它等同于哪个条目：

| 简名 | 目录条目 |
| --- | --- |
| `speed` | `nockpt-resident` |
| `balanced` | `ckpt-ring16` |
| `low` | `ckpt-ring4-actoff-optstream` |
| `minimal` | `ckpt-ring2-actoff-optstream-gradoff` |

`kohakumoss-doctor --memory` 打印针对本地设备解析后的这四个配置档。

<a id="auto"></a>
### `auto`

`auto` 从 `speed`、`balanced`、`low`、`minimal` 中选择第一个
与其他配置项兼容、且预估用量能装入规划时设备可用内存的配置档：

- 预算为可用内存乘以 0.92。CUDA 上下文已不属于可用内存，
  因此不会重复计入估算中的上下文部分。
- 规划器的逐配置档记录保存在该运行的内存报告中；每行记录
  `text_memory` 和 `estimate_with_text`。
- 使用进程组时，由 rank 0 规划，每个 rank 都解析 rank 0 的配置档
  （通过 store 传递，`kohakumoss.parallel.launch.rank0_value`），
  因此无论自身可用内存多少，每个 rank 都构建相同计划。
- 没有配置档的估算能装下时，`auto` 会停止运行，并报错列出每个配置档的估算和可用内存。
  `MEMORY_AUTO_FORCE=True` 会改为运行最后一个可运行的配置档（通常是 `minimal`），
  并以 WARNING 记录 `MEMORY_PROFILE='auto' -> <profile>
  (nothing fits, forced): no profile's estimate fits the free memory; MEMORY_AUTO_FORCE
  runs it anyway`。

在训练设备上在线编码文本时，每个配置档的估算都包含文本编码器，
使用能与该配置档的 DiT 步骤一起装入预算的最快放置方式（优先 `resident`，否则 `ring`），
或显式设置的 `TEXT_ENCODE['memory']`。其大小仅根据 `config.json` 得出，
使用计划中的数据类型和精度构造一个无权重的 meta 设备孪生模型
（`kohakumoss.training.build.text_estimate`），按一次处理
`TEXT_ENCODE['max_tokens']`（默认 16384）个词元计算；规划前不读取任何权重。
卸载 / 环形流式处理的峰值叠加在整个运行期间都分配的部分之上
（上下文、权重、适配器、优化器状态和槽位、状态环与梯度环、EMA）。

这些估算是解析式的：它们建模已分配字节数加上碎片系数，而不是设备使用量实测值
（[预估显存峰值](#estimated-peak-vram)）。

<a id="encoder-placement"></a>
## 编码器放置

训练编码器遵循 `TEXT_ENCODE["memory"]` 和 `VAE_ENCODE["memory"]`，
它们始终优先生效（DiT 常驻时也可以单独卸载编码器）。
当字典未设置 `memory` 时：

| 编码器 | 未设置 `memory` 时的放置 |
| --- | --- |
| 文本编码器 | 所有计划下都为 `auto`：选择能装入 DiT 步骤之外剩余内存的最快放置方式（优先 `resident`，否则 `ring`），因此大显卡上无论 DiT 是否流式传输，文本编码器都会常驻 |
| VAE | 遵循解析后的内存计划，无论是通过何种名称选择它（简名、目录条目、`auto` 或 `manual`）：DiT 块流式传输时卸载，否则常驻（`kohakumoss.training.build.model.encoder_memory`） |

构建会记录每项放置及其来源，例如 `text encoder placement: resident
(TEXT_ENCODE['memory'] unset: the fastest placement that fits)`。
显存配置层会显式设置这两个字典（[configs.md](../guides/configs.md#layers)）。

<a id="releasing-cached-memory"></a>
## 释放缓存内存

`STEP_RELEASE` 在训练循环内将空闲的缓存设备内存归还驱动程序：

| 值 | 释放时机 |
| --- | --- |
| `"none"`（默认） | 分配器在步骤之间保留其内存块 |
| `"empty_cache"` | 每次优化器更新之后 |
| 整数 N | 每 N 步 |

释放会带来一次设备同步和下一步的重新分配，
以换取小显卡上更低的 nvidia-smi 稳态占用。

<a id="memory-knobs"></a>
## 内存配置项

| 配置项 | 计划字段 | 值 | 默认值 | 效果 |
| --- | --- | --- | --- | --- |
| `CHECKPOINT` | `checkpoint` | `none`, `block`, `sub-block`, `selective`, `partial:<fraction>` | `block` | 反向期间重新计算哪些激活 |
| `STREAM` | `stream` | `off`、`ring`（冻结，仅主机到设备）、`swap`（可训练） | `off` | 块权重位于主机并流式传入 |
| `STREAM_RESIDENT` | `stream_resident` | int >= 2 | `0`（`STREAM='off'` 时不使用） | 流式传输时保留在设备上的块数 |
| `PINNED` | `pinned` | bool | `True` | 使用锁页还是暂存主机缓冲区（流式传输、卸载、主机优化器状态） |
| `ACT_OFFLOAD` | `act_offload` | bool | `False` | 保存的激活进入锁页主机内存，提前一个块预取 |
| `OPTIM_STATE` | `optim_state` | `gpu`、`stream`（主机状态、GPU 上更新）、`cpu`（主机上更新） | `gpu` | 仅控制 AdamW 状态放置（`OPTIMIZER='adamw'`；其他优化器保持 `gpu`） |
| `GRAD_OFFLOAD` | `grad_offload` | bool | `False` | 在主机内存中对 GRAD_ACC 的梯度求和 |
| `FUSED_STEP` | `fused_step` | bool | `False` | 在反向内部执行逐参数优化器更新 |

推理（`sample.py`）没有反向，因此那里只有 `STREAM`、`STREAM_RESIDENT` 和 `PINNED`；
其计划绝不会使用梯度检查点或卸载激活，`swap` 会变为 `ring`。

`compile` 和 `kernels` 是分别归 `COMPILE` 和 `KERNELS` 配置项控制的计划字段。
`act_offload_inflight`（int >= 1，默认 2；仅通过 `MEMORY_OVERRIDES` 设置）
决定主机等待最早一个块的激活卸载复制完成之前，最多可以排队多少个块的复制
（见 [offload.md](../internals/offload.md#activation-offload)）。

<a id="precision-knobs"></a>
## 精度配置项

精度与配置档分开设置：这些配置项在所有配置档及手动模式下都生效，
而且只有 `BASE_PRECISION` 用于选择量化基座：

| 配置项 | 计划字段 | 值 | 默认值 | 效果 |
| --- | --- | --- | --- | --- |
| `BASE_PRECISION` | `base_precision` | `auto`、`compute`、`fp8`、`int8`、`quant`（架构具备 fp8 GEMM 时使用 fp8，否则 int8）、`fp8_compute`、`mxfp8` | `auto` | 冻结基座权重的存储方式（`auto`：DiT 文件所存储的格式，16 位文件为 PRECISION 数据类型，预量化文件为其自身的基座；`compute`：PRECISION 数据类型，预量化文件则反量化；`fp8` / `int8` 仅量化权重）；`fp8_compute` / `mxfp8` 在 sm89+ 上以 fp8 执行 GEMM；参见 [quant-kernels.md](../internals/quant-kernels.md#stored-formats) |
| `FP8_RECIPE` | `fp8_recipe` | `row`, `tensor`, `block` | `row` | `fp8_compute` 的缩放粒度 |
| `FP8_DGRAD` | `fp8_dgrad` | `fp8`, `16bit` | `fp8` | 输入梯度 GEMM 使用 fp8（保留一份 W^T fp8 副本，每个权重共 2 字节）或 16 位（每个权重 1 字节） |

<a id="rules-that-constrain-a-plan"></a>
## 约束计划的规则

对每种基座精度，权重流式传输都不需要梯度检查点：流式线性层只保存其输入
（冻结基座上的 fp8 计算线性层什么也不保存），其反向会再次流式传入该块权重
（[offload.md](../internals/offload.md#weight-streaming)）。
其余规则源于实际的物理约束：

| 规则 | 原因 |
| --- | --- |
| `stream_resident >= 2` | 一个槽位计算时，下一个槽位加载 |
| `stream='ring'` 仅适用于冻结基座；`swap` 仅适用于可训练基座 | 环形槽位覆盖时不会回写；`swap` 会将训练后的块写回主机 |
| `swap`：GPU `optim_state` 时 `fused_step` 为 `True`，`optim_state='cpu'` 时为 `False` | 交换的块在其反向后离开设备，因此 GPU 更新在反向内部运行；主机更新在反向之后运行 |
| 量化基座（`fp8`、`int8`、`quant`）不可训练 | 存储的权重不是训练所使用的值 |
| `fp8_compute` / `mxfp8` 全量微调保持权重常驻 | 每步之后都要重新量化 fp8 副本 |
| `fp8`、`fp8_compute`、`mxfp8` 需要 sm89+（CUDA） | fp8 GEMM |
| 非 `gpu` 的 `optim_state` 需要 `OPTIMIZER='adamw'` | 它用于放置 AdamW 状态 |
| `swap` 需要 `OPTIMIZER='adamw'`；Muon 需要 `fused_step=False` | Muon 在反向后更新完整矩阵（使用 Muon 的全量微调采用常驻、非融合配置档，例如 `ckpt-resident`） |
| DDP 支持所有配置档；FSDP / pipeline / threaded 需要常驻的计算精度权重、GPU 状态，无激活卸载、无 `fused_step`、无 `grad_offload` | 无论计划把梯度移到哪里，DDP 都会跨 rank 对每个梯度求和；其他多 GPU 策略自行管理参数；参见 [multi-gpu.md](../guides/multi-gpu.md) |
| `checkpoint='sub-block'` 不可与 `COMPILE='block'` 或 FSDP 配合 | 子块梯度检查点直接调用块的组成部分 |

全量微调（`ALGO='full'`）时，目录条目的 `ring` 解析为 `swap`，
融合更新遵循上述 `swap` 规则。

<a id="the-catalog"></a>
## 配置档目录

目录是五个放置轴的完整笛卡尔积，每个条目按其非空部分命名：

| 轴 | 组成部分 |
| --- | --- |
| 梯度检查点 | `nockpt`（无）、`ckpt`（块）、`subckpt`（子块） |
| 权重 | `resident`、`ring16`、`ring8`、`ring4`、`ring2`（环形流式传输，对应数量的常驻块；`stream16` .. `stream2` 也被接受，表示相同部分） |
| 激活 | （无）、`actoff` |
| 优化器 | （gpu）、`optfused`（gpu，融合更新）、`optstream`（融合）、`optcpu` |
| 梯度 | （无）、`gradoff` |

3 x 5 x 2 x 4 x 2 = 240 个条目（8 位状态由 `OPTIMIZER` 选择，
为 `adamw8bit` 或 `adamw8bit_kernel`，支持任意 GPU 放置，不属于放置轴）；
对于计算精度的冻结基座，这些轴的每种组合都能通过验证，因此没有排除任何组合。
简名（[选择配置档](#choosing-a-profile)）就是其中四个条目。

<!-- catalog:begin -->
| 配置档 | 梯度检查点 | 权重 | 激活卸载 | 优化器 | 梯度卸载 | LoRA GiB | 全量微调 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `speed` | none | resident | off | gpu | off | 35.0 | 106.0 |
| `balanced` | block | ring 16 | off | gpu | off | 8.2 | 53.0 |
| `low` | block | ring 4 | on | stream fused | off | 3.9 | 13.7 |
| `minimal` | block | ring 2 | on | stream fused | on | 3.3 | 14.0 |
| `nockpt-resident` | none | resident | off | gpu | off | 35.0 | 106.0 |
| `nockpt-resident-gradoff` | none | resident | off | gpu | on | 36.1 | 89.8 |
| `nockpt-resident-optfused` | none | resident | off | gpu fused | off | 35.0 | 86.0 |
| `nockpt-resident-optfused-gradoff` | none | resident | off | gpu fused | on | 36.1 | 89.8 |
| `nockpt-resident-optstream` | none | resident | off | stream fused | off | 36.0 | 49.5 |
| `nockpt-resident-optstream-gradoff` | none | resident | off | stream fused | on | 35.8 | 50.2 |
| `nockpt-resident-optcpu` | none | resident | off | cpu | off | 36.0 | 67.4 |
| `nockpt-resident-optcpu-gradoff` | none | resident | off | cpu | on | 35.8 | 47.4 |
| `nockpt-resident-actoff` | none | resident | on | gpu | off | 13.7 | 87.3 |
| `nockpt-resident-actoff-gradoff` | none | resident | on | gpu | on | 13.6 | 67.3 |
| `nockpt-resident-actoff-optfused` | none | resident | on | gpu fused | off | 13.7 | 66.6 |
| `nockpt-resident-actoff-optfused-gradoff` | none | resident | on | gpu fused | on | 13.6 | 67.3 |
| `nockpt-resident-actoff-optstream` | none | resident | on | stream fused | off | 13.4 | 27.0 |
| `nockpt-resident-actoff-optstream-gradoff` | none | resident | on | stream fused | on | 13.3 | 27.7 |
| `nockpt-resident-actoff-optcpu` | none | resident | on | cpu | off | 13.4 | 44.9 |
| `nockpt-resident-actoff-optcpu-gradoff` | none | resident | on | cpu | on | 13.3 | 24.9 |
| `nockpt-ring16` | none | ring 16 | off | gpu | off | 30.1 | 74.9 |
| `nockpt-ring16-gradoff` | none | ring 16 | off | gpu | on | 30.0 | 75.6 |
| `nockpt-ring16-optfused` | none | ring 16 | off | gpu fused | off | 30.1 | 74.9 |
| `nockpt-ring16-optfused-gradoff` | none | ring 16 | off | gpu fused | on | 30.0 | 75.6 |
| `nockpt-ring16-optstream` | none | ring 16 | off | stream fused | off | 29.8 | 39.6 |
| `nockpt-ring16-optstream-gradoff` | none | ring 16 | off | stream fused | on | 29.6 | 40.3 |
| `nockpt-ring16-optcpu` | none | ring 16 | off | cpu | off | 29.8 | 30.4 |
| `nockpt-ring16-optcpu-gradoff` | none | ring 16 | off | cpu | on | 29.6 | 31.1 |
| `nockpt-ring16-actoff` | none | ring 16 | on | gpu | off | 7.6 | 52.4 |
| `nockpt-ring16-actoff-gradoff` | none | ring 16 | on | gpu | on | 7.4 | 53.1 |
| `nockpt-ring16-actoff-optfused` | none | ring 16 | on | gpu fused | off | 7.6 | 52.4 |
| `nockpt-ring16-actoff-optfused-gradoff` | none | ring 16 | on | gpu fused | on | 7.4 | 53.1 |
| `nockpt-ring16-actoff-optstream` | none | ring 16 | on | stream fused | off | 7.3 | 17.1 |
| `nockpt-ring16-actoff-optstream-gradoff` | none | ring 16 | on | stream fused | on | 7.1 | 17.8 |
| `nockpt-ring16-actoff-optcpu` | none | ring 16 | on | cpu | off | 7.3 | 7.9 |
| `nockpt-ring16-actoff-optcpu-gradoff` | none | ring 16 | on | cpu | on | 7.1 | 8.6 |
| `nockpt-ring8` | none | ring 8 | off | gpu | off | 28.2 | 73.0 |
| `nockpt-ring8-gradoff` | none | ring 8 | off | gpu | on | 28.1 | 73.8 |
| `nockpt-ring8-optfused` | none | ring 8 | off | gpu fused | off | 28.2 | 73.0 |
| `nockpt-ring8-optfused-gradoff` | none | ring 8 | off | gpu fused | on | 28.1 | 73.8 |
| `nockpt-ring8-optstream` | none | ring 8 | off | stream fused | off | 27.9 | 37.7 |
| `nockpt-ring8-optstream-gradoff` | none | ring 8 | off | stream fused | on | 27.8 | 38.4 |
| `nockpt-ring8-optcpu` | none | ring 8 | off | cpu | off | 27.9 | 28.5 |
| `nockpt-ring8-optcpu-gradoff` | none | ring 8 | off | cpu | on | 27.8 | 29.2 |
| `nockpt-ring8-actoff` | none | ring 8 | on | gpu | off | 5.7 | 50.5 |
| `nockpt-ring8-actoff-gradoff` | none | ring 8 | on | gpu | on | 5.5 | 51.2 |
| `nockpt-ring8-actoff-optfused` | none | ring 8 | on | gpu fused | off | 5.7 | 50.5 |
| `nockpt-ring8-actoff-optfused-gradoff` | none | ring 8 | on | gpu fused | on | 5.5 | 51.2 |
| `nockpt-ring8-actoff-optstream` | none | ring 8 | on | stream fused | off | 5.4 | 15.2 |
| `nockpt-ring8-actoff-optstream-gradoff` | none | ring 8 | on | stream fused | on | 5.2 | 15.9 |
| `nockpt-ring8-actoff-optcpu` | none | ring 8 | on | cpu | off | 5.4 | 6.0 |
| `nockpt-ring8-actoff-optcpu-gradoff` | none | ring 8 | on | cpu | on | 5.2 | 6.7 |
| `nockpt-ring4` | none | ring 4 | off | gpu | off | 27.3 | 72.1 |
| `nockpt-ring4-gradoff` | none | ring 4 | off | gpu | on | 27.1 | 72.8 |
| `nockpt-ring4-optfused` | none | ring 4 | off | gpu fused | off | 27.3 | 72.1 |
| `nockpt-ring4-optfused-gradoff` | none | ring 4 | off | gpu fused | on | 27.1 | 72.8 |
| `nockpt-ring4-optstream` | none | ring 4 | off | stream fused | off | 27.0 | 36.8 |
| `nockpt-ring4-optstream-gradoff` | none | ring 4 | off | stream fused | on | 26.8 | 37.5 |
| `nockpt-ring4-optcpu` | none | ring 4 | off | cpu | off | 27.0 | 27.6 |
| `nockpt-ring4-optcpu-gradoff` | none | ring 4 | off | cpu | on | 26.8 | 28.3 |
| `nockpt-ring4-actoff` | none | ring 4 | on | gpu | off | 4.8 | 49.6 |
| `nockpt-ring4-actoff-gradoff` | none | ring 4 | on | gpu | on | 4.6 | 50.3 |
| `nockpt-ring4-actoff-optfused` | none | ring 4 | on | gpu fused | off | 4.8 | 49.6 |
| `nockpt-ring4-actoff-optfused-gradoff` | none | ring 4 | on | gpu fused | on | 4.6 | 50.3 |
| `nockpt-ring4-actoff-optstream` | none | ring 4 | on | stream fused | off | 4.4 | 14.2 |
| `nockpt-ring4-actoff-optstream-gradoff` | none | ring 4 | on | stream fused | on | 4.3 | 14.9 |
| `nockpt-ring4-actoff-optcpu` | none | ring 4 | on | cpu | off | 4.4 | 5.0 |
| `nockpt-ring4-actoff-optcpu-gradoff` | none | ring 4 | on | cpu | on | 4.3 | 5.7 |
| `nockpt-ring2` | none | ring 2 | off | gpu | off | 26.8 | 71.6 |
| `nockpt-ring2-gradoff` | none | ring 2 | off | gpu | on | 26.7 | 72.3 |
| `nockpt-ring2-optfused` | none | ring 2 | off | gpu fused | off | 26.8 | 71.6 |
| `nockpt-ring2-optfused-gradoff` | none | ring 2 | off | gpu fused | on | 26.7 | 72.3 |
| `nockpt-ring2-optstream` | none | ring 2 | off | stream fused | off | 26.5 | 36.3 |
| `nockpt-ring2-optstream-gradoff` | none | ring 2 | off | stream fused | on | 26.3 | 37.0 |
| `nockpt-ring2-optcpu` | none | ring 2 | off | cpu | off | 26.5 | 27.1 |
| `nockpt-ring2-optcpu-gradoff` | none | ring 2 | off | cpu | on | 26.3 | 27.8 |
| `nockpt-ring2-actoff` | none | ring 2 | on | gpu | off | 4.3 | 49.1 |
| `nockpt-ring2-actoff-gradoff` | none | ring 2 | on | gpu | on | 4.1 | 49.8 |
| `nockpt-ring2-actoff-optfused` | none | ring 2 | on | gpu fused | off | 4.3 | 49.1 |
| `nockpt-ring2-actoff-optfused-gradoff` | none | ring 2 | on | gpu fused | on | 4.1 | 49.8 |
| `nockpt-ring2-actoff-optstream` | none | ring 2 | on | stream fused | off | 4.0 | 13.8 |
| `nockpt-ring2-actoff-optstream-gradoff` | none | ring 2 | on | stream fused | on | 3.8 | 14.5 |
| `nockpt-ring2-actoff-optcpu` | none | ring 2 | on | cpu | off | 4.0 | 4.6 |
| `nockpt-ring2-actoff-optcpu-gradoff` | none | ring 2 | on | cpu | on | 3.8 | 5.3 |
| `ckpt-resident` | block | resident | off | gpu | off | 13.9 | 84.8 |
| `ckpt-resident-gradoff` | block | resident | off | gpu | on | 14.2 | 67.9 |
| `ckpt-resident-optfused` | block | resident | off | gpu fused | off | 13.9 | 64.8 |
| `ckpt-resident-optfused-gradoff` | block | resident | off | gpu fused | on | 14.2 | 67.9 |
| `ckpt-resident-optstream` | block | resident | off | stream fused | off | 14.0 | 27.6 |
| `ckpt-resident-optstream-gradoff` | block | resident | off | stream fused | on | 13.9 | 28.3 |
| `ckpt-resident-optcpu` | block | resident | off | cpu | off | 14.0 | 45.5 |
| `ckpt-resident-optcpu-gradoff` | block | resident | off | cpu | on | 13.9 | 25.5 |
| `ckpt-resident-actoff` | block | resident | on | gpu | off | 13.2 | 86.8 |
| `ckpt-resident-actoff-gradoff` | block | resident | on | gpu | on | 13.1 | 66.8 |
| `ckpt-resident-actoff-optfused` | block | resident | on | gpu fused | off | 13.2 | 66.1 |
| `ckpt-resident-actoff-optfused-gradoff` | block | resident | on | gpu fused | on | 13.1 | 66.8 |
| `ckpt-resident-actoff-optstream` | block | resident | on | stream fused | off | 12.9 | 26.5 |
| `ckpt-resident-actoff-optstream-gradoff` | block | resident | on | stream fused | on | 12.7 | 27.2 |
| `ckpt-resident-actoff-optcpu` | block | resident | on | cpu | off | 12.9 | 44.4 |
| `ckpt-resident-actoff-optcpu-gradoff` | block | resident | on | cpu | on | 12.7 | 24.4 |
| `ckpt-ring16` | block | ring 16 | off | gpu | off | 8.2 | 53.0 |
| `ckpt-ring16-gradoff` | block | ring 16 | off | gpu | on | 8.1 | 53.7 |
| `ckpt-ring16-optfused` | block | ring 16 | off | gpu fused | off | 8.2 | 53.0 |
| `ckpt-ring16-optfused-gradoff` | block | ring 16 | off | gpu fused | on | 8.1 | 53.7 |
| `ckpt-ring16-optstream` | block | ring 16 | off | stream fused | off | 7.9 | 17.7 |
| `ckpt-ring16-optstream-gradoff` | block | ring 16 | off | stream fused | on | 7.7 | 18.4 |
| `ckpt-ring16-optcpu` | block | ring 16 | off | cpu | off | 7.9 | 8.5 |
| `ckpt-ring16-optcpu-gradoff` | block | ring 16 | off | cpu | on | 7.7 | 9.2 |
| `ckpt-ring16-actoff` | block | ring 16 | on | gpu | off | 7.1 | 51.9 |
| `ckpt-ring16-actoff-gradoff` | block | ring 16 | on | gpu | on | 6.9 | 52.6 |
| `ckpt-ring16-actoff-optfused` | block | ring 16 | on | gpu fused | off | 7.1 | 51.9 |
| `ckpt-ring16-actoff-optfused-gradoff` | block | ring 16 | on | gpu fused | on | 6.9 | 52.6 |
| `ckpt-ring16-actoff-optstream` | block | ring 16 | on | stream fused | off | 6.8 | 16.6 |
| `ckpt-ring16-actoff-optstream-gradoff` | block | ring 16 | on | stream fused | on | 6.6 | 17.3 |
| `ckpt-ring16-actoff-optcpu` | block | ring 16 | on | cpu | off | 6.8 | 7.4 |
| `ckpt-ring16-actoff-optcpu-gradoff` | block | ring 16 | on | cpu | on | 6.6 | 8.1 |
| `ckpt-ring8` | block | ring 8 | off | gpu | off | 6.3 | 51.1 |
| `ckpt-ring8-gradoff` | block | ring 8 | off | gpu | on | 6.2 | 51.8 |
| `ckpt-ring8-optfused` | block | ring 8 | off | gpu fused | off | 6.3 | 51.1 |
| `ckpt-ring8-optfused-gradoff` | block | ring 8 | off | gpu fused | on | 6.2 | 51.8 |
| `ckpt-ring8-optstream` | block | ring 8 | off | stream fused | off | 6.0 | 15.8 |
| `ckpt-ring8-optstream-gradoff` | block | ring 8 | off | stream fused | on | 5.8 | 16.5 |
| `ckpt-ring8-optcpu` | block | ring 8 | off | cpu | off | 6.0 | 6.6 |
| `ckpt-ring8-optcpu-gradoff` | block | ring 8 | off | cpu | on | 5.8 | 7.3 |
| `ckpt-ring8-actoff` | block | ring 8 | on | gpu | off | 5.2 | 50.0 |
| `ckpt-ring8-actoff-gradoff` | block | ring 8 | on | gpu | on | 5.0 | 50.7 |
| `ckpt-ring8-actoff-optfused` | block | ring 8 | on | gpu fused | off | 5.2 | 50.0 |
| `ckpt-ring8-actoff-optfused-gradoff` | block | ring 8 | on | gpu fused | on | 5.0 | 50.7 |
| `ckpt-ring8-actoff-optstream` | block | ring 8 | on | stream fused | off | 4.9 | 14.7 |
| `ckpt-ring8-actoff-optstream-gradoff` | block | ring 8 | on | stream fused | on | 4.7 | 15.4 |
| `ckpt-ring8-actoff-optcpu` | block | ring 8 | on | cpu | off | 4.9 | 5.5 |
| `ckpt-ring8-actoff-optcpu-gradoff` | block | ring 8 | on | cpu | on | 4.7 | 6.2 |
| `ckpt-ring4` | block | ring 4 | off | gpu | off | 5.4 | 50.2 |
| `ckpt-ring4-gradoff` | block | ring 4 | off | gpu | on | 5.2 | 50.9 |
| `ckpt-ring4-optfused` | block | ring 4 | off | gpu fused | off | 5.4 | 50.2 |
| `ckpt-ring4-optfused-gradoff` | block | ring 4 | off | gpu fused | on | 5.2 | 50.9 |
| `ckpt-ring4-optstream` | block | ring 4 | off | stream fused | off | 5.1 | 14.9 |
| `ckpt-ring4-optstream-gradoff` | block | ring 4 | off | stream fused | on | 4.9 | 15.6 |
| `ckpt-ring4-optcpu` | block | ring 4 | off | cpu | off | 5.1 | 5.6 |
| `ckpt-ring4-optcpu-gradoff` | block | ring 4 | off | cpu | on | 4.9 | 6.4 |
| `ckpt-ring4-actoff` | block | ring 4 | on | gpu | off | 4.3 | 49.1 |
| `ckpt-ring4-actoff-gradoff` | block | ring 4 | on | gpu | on | 4.1 | 49.8 |
| `ckpt-ring4-actoff-optfused` | block | ring 4 | on | gpu fused | off | 4.3 | 49.1 |
| `ckpt-ring4-actoff-optfused-gradoff` | block | ring 4 | on | gpu fused | on | 4.1 | 49.8 |
| `ckpt-ring4-actoff-optstream` | block | ring 4 | on | stream fused | off | 3.9 | 13.7 |
| `ckpt-ring4-actoff-optstream-gradoff` | block | ring 4 | on | stream fused | on | 3.8 | 14.4 |
| `ckpt-ring4-actoff-optcpu` | block | ring 4 | on | cpu | off | 3.9 | 4.5 |
| `ckpt-ring4-actoff-optcpu-gradoff` | block | ring 4 | on | cpu | on | 3.8 | 5.2 |
| `ckpt-ring2` | block | ring 2 | off | gpu | off | 4.9 | 49.7 |
| `ckpt-ring2-gradoff` | block | ring 2 | off | gpu | on | 4.7 | 50.4 |
| `ckpt-ring2-optfused` | block | ring 2 | off | gpu fused | off | 4.9 | 49.7 |
| `ckpt-ring2-optfused-gradoff` | block | ring 2 | off | gpu fused | on | 4.7 | 50.4 |
| `ckpt-ring2-optstream` | block | ring 2 | off | stream fused | off | 4.6 | 14.4 |
| `ckpt-ring2-optstream-gradoff` | block | ring 2 | off | stream fused | on | 4.4 | 15.1 |
| `ckpt-ring2-optcpu` | block | ring 2 | off | cpu | off | 4.6 | 5.2 |
| `ckpt-ring2-optcpu-gradoff` | block | ring 2 | off | cpu | on | 4.4 | 5.9 |
| `ckpt-ring2-actoff` | block | ring 2 | on | gpu | off | 3.8 | 48.6 |
| `ckpt-ring2-actoff-gradoff` | block | ring 2 | on | gpu | on | 3.6 | 49.3 |
| `ckpt-ring2-actoff-optfused` | block | ring 2 | on | gpu fused | off | 3.8 | 48.6 |
| `ckpt-ring2-actoff-optfused-gradoff` | block | ring 2 | on | gpu fused | on | 3.6 | 49.3 |
| `ckpt-ring2-actoff-optstream` | block | ring 2 | on | stream fused | off | 3.5 | 13.3 |
| `ckpt-ring2-actoff-optstream-gradoff` | block | ring 2 | on | stream fused | on | 3.3 | 14.0 |
| `ckpt-ring2-actoff-optcpu` | block | ring 2 | on | cpu | off | 3.5 | 4.0 |
| `ckpt-ring2-actoff-optcpu-gradoff` | block | ring 2 | on | cpu | on | 3.3 | 4.8 |
| `subckpt-resident` | sub-block | resident | off | gpu | off | 14.7 | 85.7 |
| `subckpt-resident-gradoff` | sub-block | resident | off | gpu | on | 15.1 | 68.8 |
| `subckpt-resident-optfused` | sub-block | resident | off | gpu fused | off | 14.7 | 65.7 |
| `subckpt-resident-optfused-gradoff` | sub-block | resident | off | gpu fused | on | 15.1 | 68.8 |
| `subckpt-resident-optstream` | sub-block | resident | off | stream fused | off | 14.9 | 28.5 |
| `subckpt-resident-optstream-gradoff` | sub-block | resident | off | stream fused | on | 14.8 | 29.2 |
| `subckpt-resident-optcpu` | sub-block | resident | off | cpu | off | 14.9 | 46.4 |
| `subckpt-resident-optcpu-gradoff` | sub-block | resident | off | cpu | on | 14.8 | 26.4 |
| `subckpt-resident-actoff` | sub-block | resident | on | gpu | off | 13.0 | 86.6 |
| `subckpt-resident-actoff-gradoff` | sub-block | resident | on | gpu | on | 12.8 | 66.5 |
| `subckpt-resident-actoff-optfused` | sub-block | resident | on | gpu fused | off | 13.0 | 65.8 |
| `subckpt-resident-actoff-optfused-gradoff` | sub-block | resident | on | gpu fused | on | 12.8 | 66.5 |
| `subckpt-resident-actoff-optstream` | sub-block | resident | on | stream fused | off | 12.7 | 26.3 |
| `subckpt-resident-actoff-optstream-gradoff` | sub-block | resident | on | stream fused | on | 12.5 | 27.0 |
| `subckpt-resident-actoff-optcpu` | sub-block | resident | on | cpu | off | 12.7 | 44.2 |
| `subckpt-resident-actoff-optcpu-gradoff` | sub-block | resident | on | cpu | on | 12.5 | 24.1 |
| `subckpt-ring16` | sub-block | ring 16 | off | gpu | off | 9.1 | 53.9 |
| `subckpt-ring16-gradoff` | sub-block | ring 16 | off | gpu | on | 9.0 | 54.6 |
| `subckpt-ring16-optfused` | sub-block | ring 16 | off | gpu fused | off | 9.1 | 53.9 |
| `subckpt-ring16-optfused-gradoff` | sub-block | ring 16 | off | gpu fused | on | 9.0 | 54.6 |
| `subckpt-ring16-optstream` | sub-block | ring 16 | off | stream fused | off | 8.8 | 18.6 |
| `subckpt-ring16-optstream-gradoff` | sub-block | ring 16 | off | stream fused | on | 8.6 | 19.3 |
| `subckpt-ring16-optcpu` | sub-block | ring 16 | off | cpu | off | 8.8 | 9.4 |
| `subckpt-ring16-optcpu-gradoff` | sub-block | ring 16 | off | cpu | on | 8.6 | 10.1 |
| `subckpt-ring16-actoff` | sub-block | ring 16 | on | gpu | off | 6.9 | 51.7 |
| `subckpt-ring16-actoff-gradoff` | sub-block | ring 16 | on | gpu | on | 6.7 | 52.4 |
| `subckpt-ring16-actoff-optfused` | sub-block | ring 16 | on | gpu fused | off | 6.9 | 51.7 |
| `subckpt-ring16-actoff-optfused-gradoff` | sub-block | ring 16 | on | gpu fused | on | 6.7 | 52.4 |
| `subckpt-ring16-actoff-optstream` | sub-block | ring 16 | on | stream fused | off | 6.5 | 16.3 |
| `subckpt-ring16-actoff-optstream-gradoff` | sub-block | ring 16 | on | stream fused | on | 6.4 | 17.1 |
| `subckpt-ring16-actoff-optcpu` | sub-block | ring 16 | on | cpu | off | 6.5 | 7.1 |
| `subckpt-ring16-actoff-optcpu-gradoff` | sub-block | ring 16 | on | cpu | on | 6.4 | 7.8 |
| `subckpt-ring8` | sub-block | ring 8 | off | gpu | off | 7.2 | 52.0 |
| `subckpt-ring8-gradoff` | sub-block | ring 8 | off | gpu | on | 7.1 | 52.7 |
| `subckpt-ring8-optfused` | sub-block | ring 8 | off | gpu fused | off | 7.2 | 52.0 |
| `subckpt-ring8-optfused-gradoff` | sub-block | ring 8 | off | gpu fused | on | 7.1 | 52.7 |
| `subckpt-ring8-optstream` | sub-block | ring 8 | off | stream fused | off | 6.9 | 16.7 |
| `subckpt-ring8-optstream-gradoff` | sub-block | ring 8 | off | stream fused | on | 6.7 | 17.4 |
| `subckpt-ring8-optcpu` | sub-block | ring 8 | off | cpu | off | 6.9 | 7.5 |
| `subckpt-ring8-optcpu-gradoff` | sub-block | ring 8 | off | cpu | on | 6.7 | 8.2 |
| `subckpt-ring8-actoff` | sub-block | ring 8 | on | gpu | off | 5.0 | 49.8 |
| `subckpt-ring8-actoff-gradoff` | sub-block | ring 8 | on | gpu | on | 4.8 | 50.5 |
| `subckpt-ring8-actoff-optfused` | sub-block | ring 8 | on | gpu fused | off | 5.0 | 49.8 |
| `subckpt-ring8-actoff-optfused-gradoff` | sub-block | ring 8 | on | gpu fused | on | 4.8 | 50.5 |
| `subckpt-ring8-actoff-optstream` | sub-block | ring 8 | on | stream fused | off | 4.6 | 14.5 |
| `subckpt-ring8-actoff-optstream-gradoff` | sub-block | ring 8 | on | stream fused | on | 4.5 | 15.2 |
| `subckpt-ring8-actoff-optcpu` | sub-block | ring 8 | on | cpu | off | 4.6 | 5.2 |
| `subckpt-ring8-actoff-optcpu-gradoff` | sub-block | ring 8 | on | cpu | on | 4.5 | 5.9 |
| `subckpt-ring4` | sub-block | ring 4 | off | gpu | off | 6.3 | 51.1 |
| `subckpt-ring4-gradoff` | sub-block | ring 4 | off | gpu | on | 6.1 | 51.8 |
| `subckpt-ring4-optfused` | sub-block | ring 4 | off | gpu fused | off | 6.3 | 51.1 |
| `subckpt-ring4-optfused-gradoff` | sub-block | ring 4 | off | gpu fused | on | 6.1 | 51.8 |
| `subckpt-ring4-optstream` | sub-block | ring 4 | off | stream fused | off | 6.0 | 15.8 |
| `subckpt-ring4-optstream-gradoff` | sub-block | ring 4 | off | stream fused | on | 5.8 | 16.5 |
| `subckpt-ring4-optcpu` | sub-block | ring 4 | off | cpu | off | 6.0 | 6.5 |
| `subckpt-ring4-optcpu-gradoff` | sub-block | ring 4 | off | cpu | on | 5.8 | 7.3 |
| `subckpt-ring4-actoff` | sub-block | ring 4 | on | gpu | off | 4.0 | 48.8 |
| `subckpt-ring4-actoff-gradoff` | sub-block | ring 4 | on | gpu | on | 3.9 | 49.5 |
| `subckpt-ring4-actoff-optfused` | sub-block | ring 4 | on | gpu fused | off | 4.0 | 48.8 |
| `subckpt-ring4-actoff-optfused-gradoff` | sub-block | ring 4 | on | gpu fused | on | 3.9 | 49.5 |
| `subckpt-ring4-actoff-optstream` | sub-block | ring 4 | on | stream fused | off | 3.7 | 13.5 |
| `subckpt-ring4-actoff-optstream-gradoff` | sub-block | ring 4 | on | stream fused | on | 3.5 | 14.2 |
| `subckpt-ring4-actoff-optcpu` | sub-block | ring 4 | on | cpu | off | 3.7 | 4.3 |
| `subckpt-ring4-actoff-optcpu-gradoff` | sub-block | ring 4 | on | cpu | on | 3.5 | 5.0 |
| `subckpt-ring2` | sub-block | ring 2 | off | gpu | off | 5.8 | 50.6 |
| `subckpt-ring2-gradoff` | sub-block | ring 2 | off | gpu | on | 5.6 | 51.3 |
| `subckpt-ring2-optfused` | sub-block | ring 2 | off | gpu fused | off | 5.8 | 50.6 |
| `subckpt-ring2-optfused-gradoff` | sub-block | ring 2 | off | gpu fused | on | 5.6 | 51.3 |
| `subckpt-ring2-optstream` | sub-block | ring 2 | off | stream fused | off | 5.5 | 15.3 |
| `subckpt-ring2-optstream-gradoff` | sub-block | ring 2 | off | stream fused | on | 5.3 | 16.0 |
| `subckpt-ring2-optcpu` | sub-block | ring 2 | off | cpu | off | 5.5 | 6.1 |
| `subckpt-ring2-optcpu-gradoff` | sub-block | ring 2 | off | cpu | on | 5.3 | 6.8 |
| `subckpt-ring2-actoff` | sub-block | ring 2 | on | gpu | off | 3.6 | 48.4 |
| `subckpt-ring2-actoff-gradoff` | sub-block | ring 2 | on | gpu | on | 3.4 | 49.1 |
| `subckpt-ring2-actoff-optfused` | sub-block | ring 2 | on | gpu fused | off | 3.6 | 48.4 |
| `subckpt-ring2-actoff-optfused-gradoff` | sub-block | ring 2 | on | gpu fused | on | 3.4 | 49.1 |
| `subckpt-ring2-actoff-optstream` | sub-block | ring 2 | on | stream fused | off | 3.2 | 13.0 |
| `subckpt-ring2-actoff-optstream-gradoff` | sub-block | ring 2 | on | stream fused | on | 3.1 | 13.7 |
| `subckpt-ring2-actoff-optcpu` | sub-block | ring 2 | on | cpu | off | 3.2 | 3.8 |
| `subckpt-ring2-actoff-optcpu-gradoff` | sub-block | ring 2 | on | cpu | on | 3.1 | 4.5 |
<!-- catalog:end -->

<a id="estimated-peak-vram"></a>
## 预估显存峰值

解析内存模型（`kohakumoss.memory.estimate`）针对 MOSS 基础 DiT 在 1 MP 下
（4096 图像 + 300 文本词元、bf16、融合内核）的一个微步骤：
每个块线性层上秩为 16 的 LoRA，以及全量微调（此时 `ring N` 为 `swap`）。
不包含文本编码器和 VAE（已缓存或单独放置）。`kohakumoss-config list` 打印相同表格；
`tests/memory/profiles/test_catalog_table.py` 保证两者一致。
全量微调的设备参数就是其主副本：fp32（`adamw16` 下为计算数据类型），
每个常驻可训练参数 4 B，没有独立的主副本，其梯度也是如此；
`swap` 块的常驻槽位和梯度使用计算数据类型。
全量微调列计入卸载优化器所分配的设备缓冲区：
`StreamedAdamW` 用于 `swap` 块的槽位环（按块大小设置）、
`HostStateAdamW` 的状态环和梯度卸载的暂存环
（两者都按最大的设备参数，即 6 D x D 的块共享调制参数设置大小）。

总量是设备级别的数字：分配器各部分之和乘以计划的碎片系数
（预留量 / 已分配量：所有缓冲区都在计算流上时为 `FRAGMENTATION` 1.08，
计划在旁路流上保存缓冲区时为 `FRAGMENTATION_OFFLOAD` 1.12：
流式权重、主机优化器状态、梯度或激活卸载），再加上 `context`
（0.88 GiB：CUDA 上下文、已加载内核和库，在分配器之外）。
实测的 `torch.cuda.max_memory_allocated` 峰值对应不含 `context` 的各部分之和。
在 `minimal` 下、使用 4 个常驻 `swap` 块的全量微调
（1 MP、bf16 下的 `ckpt-ring4-actoff-optstream-gradoff`），
各部分与代码为其分配的缓冲区对应如下：

| 部分 | GiB | `minimal` 在设备上保留的内容 |
| --- | --- | --- |
| `weights` | 2.06 | 非块参数（0.33 B，fp32 主副本）+ 权重流的 4 个常驻 `swap` 槽位（bf16） |
| `optim_slots` | 5.70 | `StreamedAdamW`：3 个槽位 x 每块参数（fp32 主副本、m、v 行；fp32 梯度和；bf16 更新后权重）= 18 B x 113 M |
| `state_ring` | 2.53 | `HostStateAdamW`：3 个槽位 x 最大设备参数（56.6 M）的 4 个 fp32 行（主副本、m、v、梯度） |
| `grad_ring` | 0.63 | 梯度卸载：最大设备参数的 3 个 fp32 暂存缓冲区 |
| `grads` | 0.21 | 一个块在进入其槽位前的 bf16 梯度 |
| `recompute` | 0.50 | 一个启用梯度检查点的块的重计算窗口 |
| `activations` / `text_refiner` / `io` | 0.23 | 2 个正在传输的卸载块的输入、文本细化器激活、步骤输入 / 输出 |
| `workspace` | 0.25 | GEMM / 内核工作区 |
| `context` | 0.88 | CUDA 上下文、内核和库（在分配器之外） |

已分配：12.11 GiB；预留（x 1.12）：13.56 GiB；设备总量：14.44 GiB。
在 RTX 3090 上测量（`scripts/bench/train/memory.py`）：已分配 11.93 GiB、
预留 13.26 GiB（已分配的 1.11 倍）、设备使用量 14.14 GiB（分配器之外 0.88 GiB）；
`tests/training/memory/test_estimate_swap_slots.py` 保证估算不低于这三个实测值。
每部分都是步骤峰值时实际存在的缓冲区；没有重复计数，也没有把 `minimal`
卸载的内容算进去（块的主副本、矩和梯度和都保留在主机；
只有上述环和非块主副本位于设备上）。
