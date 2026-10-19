[English](../../concepts/adapters.md)

<a id="adapters"></a>
# 适配器

KohakuMoss 使用 [lycoris-lora](https://github.com/KohakuBlueleaf/LyCORIS)
包训练适配器：采用其中的算法、参数布局、初始化、预设语义、保存格式和融合内核。
`kohakumoss.adapters` 是围绕它实现的自定义应用器。

<a id="knobs"></a>
## 配置项

`scripts/train/moss.py` 中的适配器配置项：

| 配置项 | 默认值 | 设置内容 |
| --- | --- | --- |
| `ALGO` | `lora` | 算法（[算法](#algorithms)）；`full` 表示全量微调 |
| `PRESET` | `attn-mlp` | 哪些线性层应用适配器（[目标选择](#targeting)） |
| `DIM`, `ALPHA` | 32, 32.0 | 秩（OFT：块大小上限）和 alpha |
| `ALGO_KWARGS` | `{}` | 额外的 LyCORIS 模块参数，例如 LoKr 的 `{"factor": 8}` |
| `DORA`, `DORA_DETACH` | False, False | 权重分解；训练时分离 DoRA 权重范数（此时不使用融合 DoRA 内核） |
| `DROPOUT` | `{"input": 0.0, "rank": 0.0, "module": 0.0}` | 三种 dropout |
| `ADAPTER_MODE` | `bypass` | `bypass`（适配器路径加到基座输出上）或 `merge`（在 W + dW 上执行一次 GEMM） |
| `LORAPLUS_RATIO` | 1.0（关闭） | 零初始化因子的 LoRA+ 学习率比例（[优化器参数组](#optimiser-groups)） |
| `TRAIN_NORMS` | False | 同时训练预设覆盖的归一化层上的增量 |
| `KERNELS["lycoris"]` | `auto` | 适配器内核算子的运行方式（[绑定](#binding)） |

<a id="algorithms"></a>
## 算法

| `ALGO` | LyCORIS 模块 | 说明 |
| --- | --- | --- |
| `lora`, `locon` | `LoConModule` | `DORA=True` 添加权重分解收尾计算 |
| `loha` | `LohaModule` | 支持 DoRA |
| `lokr` | `LokrModule` | 通过 `ALGO_KWARGS` 设置 `factor`、`decompose_both`、`full_matrix`；支持 DoRA |
| `oft` | `DiagOFTModule` | `DIM` 是块大小上限；`constraint`、`rescaled` |
| `boft` | `ButterflyOFTModule` | 每个输出宽度都需要可分解为 2 的幂次因子 |
| `glora` | `GLoRAModule` | dW = W*a1*a2 + b1*b2 |
| `ia3` | `IA3Module` | `train_on_input` |
| `dylora` | `DyLoraModule` | 训练时每步抽取一个秩前缀，评估时使用所有秩 |
| `tlora` | `TLoraModule` | 每步设置逐词元秩掩码；仅支持 `ADAPTER_MODE="bypass"` |

`ALGO="full"` 表示全量微调（[full-finetune.md](../guides/full-finetune.md)）：
训练主干自身的权重，不构建 LyCORIS 模块。

- **`ALGO_KWARGS`** 的键会根据 LyCORIS 模块（以及 `LycorisNetwork`）的构造函数
  进行检查：未知键或已有独立配置项的键（`alpha`、`lora_dim`、各类 dropout）都会报错。
- **T-LoRA 的掩码**接受 `mask_min_rank`、`mask_alpha` 和 `mask_max_timestep`
  （LyCORIS 的 `compute_timestep_mask`，其中 t = sigma * max_timestep）；
  这些参数不会传给模块。
- **DoRA** 不允许输入 dropout（LyCORIS 会忽略它）。
- **`TRAIN_NORMS=True`** 向预设覆盖的归一化层添加一个可训练增量
  （`adapters.train.norms.NormDelta`，一种参数化：`weight = original + m *
  w_norm`，以 fp32 求和，并只向归一化层的数据类型舍入一次）。

每种算法对应一条 `adapters.kinds.Kind` 记录（dW 构建器、融合与即时执行的前向工厂、
ComfyUI 转换、行缩放张量、推理堆叠类、LoRA+ 侧），按种类名、LyCORIS 模块类
和 `ALGO` 名称注册；`adapters.register(kind)` 可添加一条记录。
参见 `docs/internals/adapter-kinds.md`。

<a id="targeting"></a>
## 目标选择

`PRESET` 由 `adapters.build.presets.resolve_preset` 解析为完整的 LyCORIS 预设
（以模块路径为对象的 `target_name` 正则表达式，这些路径也就是检查点键路径）：

| 基础预设 | 层 |
| --- | --- |
| `attn` | 联合块的 `layers.N.proj.{q,k,v,o}` |
| `attn-mlp` | 再加上联合块的 `ffn_up_gate`、`ffn_down` |
| `full-block` | 再加上文本细化器块的线性层 |
| `all` | transformer 的每个线性层 |

- **修饰符**追加在后面：`+text-refiner`、`+modulation`（时间嵌入、共享调制和输出头调制）、
  `+embed`（视频嵌入器、最终线性层）。
- **自定义预设。** toml 路径，或带有任意 LyCORIS 预设键（`target_name`、
  `name_algo_map`、`module_algo_map`、`exclude_name`、`use_fnmatch`）的字典，都会原样使用；
  每次调用都会显式设置每个键，因为 `LycorisNetwork` 将预设保存在类状态中。

对于基座存放在量化或流式存储中的模型，LyCORIS 根据线性层的输入输出特征数确定每个模块的尺寸，
而构建时需要读取基座的状态（DoRA 幅度、T-LoRA SVD、OFT / BOFT 重缩放）
则从每层基座的一个临时副本中取得，每次只处理一层。

<a id="optimiser-groups"></a>
## 优化器参数组

`param_groups(lr, weight_decay, rules, use_muon)`：

- 遵循 `WD_RULES`（`moss`：对一维张量、DoRA 幅度、重缩放参数、
  标量和 T-LoRA 的 lambda 不应用衰减；`none`；`all`）；
- 在 `OPTIMIZER="muon"` 下，将两个维度都 > 1 的二维因子标记为可使用 Muon；
- 应用 LoRA+（`LORAPLUS_RATIO`）：按比例放大每种算法的零初始化因子的学习率：
  LoCon 的 `lora_up`、LoHa 的 `hada_w2_a`、LoKr 的 `lokr_w2` /
  `lokr_w2_b`、GLoRA 的 `a2` / `b2`、DyLoRA 的 up 块；以及 T-LoRA 的 `p_layer`
  （由数据初始化，位于其输出侧）。

<a id="multiplier"></a>
## 倍率

倍率缩放 dW：

| 种类 | 倍率 `m` |
| --- | --- |
| 低秩与稠密差值 | `W + m * dW` |
| DoRA | 使分解后的权重线性移向基座权重，`W + m*(W' - W)`，其中 `W' = decompose(W + dW)`，这也是 ComfyUI 应用强度的方式 |
| OFT / BOFT | 将每个旋转插值向单位矩阵，`m*R + (1 - m)*I`（LyCORIS 的函数式 `diff_weight(..., scale=m)`） |

LyCORIS 自身的 DoRA 运行时（`weight_decompose(W + dW, dora_scale, m)`，
由 `LoConModule.forward` / `get_merged_weight` 使用，因此 kohya 和 sd-webui 的
LyCORIS 加载器也使用它）改为缩放幅度：`(W + dW) * (m * (s - 1) + 1)`，
其中 s 是逐行（或逐列）比率 `dora_scale / ||W + dW||`。两者在 `m = 1` 时一致，
在任何其他强度下都不同，所以强度为 0.7 的 DoRA 文件在那里渲染的结果
会不同于 KohakuMoss 和 ComfyUI。KohakuMoss 遵循 ComfyUI，
因为其导出面向 ComfyUI（`comfy` 格式），也因为线性移动可以让所有格式的强度
遵循同一条规则（以 `diff` 写出的层、合并操作和 DoRA 内核都计算 `W + m * delta`）。

<a id="state-and-export"></a>
## 状态与导出

- **训练状态。** `state_dict()` / `save_state()` 按 LyCORIS 键保存原始参数
  （用于精确恢复）。`state_metadata()` 提供文件的元数据，由 `save_state`
  以及每个训练检查点的 `trainable.safetensors` 写入：原始状态标签、
  适配器配置（`kohakumoss_spec`），以及各层训练时使用的权重折叠
  （`kohakumoss_fold`，一条 `fold_record`：逐块缩放、via、门前缩放、eps；
  未折叠时为 null）。全量微调的检查点携带其自身的标签和折叠。
  `load_state_dict` 要求与当前对象的键完全一致。
- **导出格式。** `export(path, fmt)` 写出：

  | `fmt` | 文件 |
  | --- | --- |
  | `lycoris` | LyCORIS 自身的状态字典；带有逐行折叠缩放的层以 `diff` 写出 |
  | `comfy` | 参见 `docs/guides/export-to-comfy.md` |
  | `merged` | 一个使用裸键的 transformer 文件 |
  | `diffusers` | transformer 文件夹：分片、索引和 `config.json` |

  合并格式经过主干自身的导出状态路径
  （`kohakumoss.model.io.save.export_state`），对存储的基座反量化并撤销残差折叠；
  `row_scales` 为 `lycoris` / `comfy` 撤销折叠；`fused_qkv`
  为 `comfy` 和 `merged` 写出 ComfyUI 的融合 `proj.qkv` 布局
  （`docs/guides/export-to-comfy.md#fused-qkv`）。
- **评估模式。** 每次导出以及 `merge_into` 都在非训练模式下评估适配器
  （无秩 dropout、使用每个 DyLoRA 块、无 T-LoRA 掩码），并保持原有训练状态不变。
- **`merge_into(m)`** 以倍率 `m` 将每个线性层与归一化层的增量固化到权重中。

<a id="inference"></a>
## 推理

`load_adapters(model, [(file, multiplier), ...], mode)` 读取：

- LyCORIS 文件；
- ComfyUI 风格文件（`lora_unet_` 或 `diffusion_model.` 键；融合的 `proj.qkv` LoRA
  或 `diff` 层会被拆分为 q / k / v）；
- 训练状态。

训练状态（其配置标记在文件元数据中）首先转换为导出时写出的未折叠 LyCORIS 张量：
根据其配置在设备自身的后端上重建不带 dropout 的模块，
每层 dW 的行除以该状态训练时的折叠（使用其记录的折叠；未记录时使用当前模型自身的折叠），
而转换期间某层读取的基座是该状态自身空间中的有效权重。
随后像 LyCORIS 文件一样加载，因此在折叠模型上训练的状态能在未折叠模型上正确渲染，反之亦然。

| 模式 | 适配器应用方式 |
| --- | --- |
| `bypass` | 将每个适配器的前向堆叠到前一个适配器之后。在稠密基座上，低秩路径保持因子化；DoRA、稠密差值、DyLoRA 和倍率 != 1 的 OFT 使用一个以该层计算数据类型保存的 dW，它根据该层当前的有效权重（基座加上此前堆叠的所有适配器）计算，与 `merge` 一样 |
| `merge` | 将 dW 加入基座权重，以 fp32 求和并只舍入一次 |
| 两种模式，在量化 / 流式基座上 | 每种算法都以旁路链形式附加（请求 merge 时会回退并记录一行日志）：DoRA 只保存其逐行（或逐列）比率，由有效权重的临时副本一次性计算；不接受稠密差值文件 |

- 加载的模块运行自身的方法（LyCORIS 在每次调用时选择后端）。
- `remove()` 逐位精确恢复每个改动过的权重和前向；`undoable=False`
  不保留副本（单向合并，正如 `scripts/export/merge.py` 所做的那样）。
- `row_scales` 将未折叠文件放置到折叠模型上。

<a id="binding"></a>
## 绑定

`apply(model, config)`（config 是 `AdapterConfig`，由 `config_from_knobs` 从配置项构造）
在匹配的线性层上构建 LyCORIS 模块（目标层为 `MossLinear`，一种 `nn.Linear`），
将每个模块注册为对应线性层的子模块 `lyco`（FSDP 分片和权重流式传输会将它与该层一起处理），
并一次性安装每层的前向。每层绑定的前向为下列之一：

- **该算法种类的内核算子：** `bypass` 将适配器路径加到基座输出上，`merge`
  构建 dW 并在 W + dW 上运行一次 GEMM，以 fp32 求和并只舍入一次；
- **DoRA 路径**，用于分解类算法：W' = DoRA(W + dW)，
  其中分解步骤是一个权重分解内核算子；
- **LyCORIS 模块的方法：** 用于带输入或秩 dropout 的训练（内核算子不支持这两者）、
  带 `DORA_DETACH` 的 DoRA 训练、GLoRA 旁路、倍率 != 1 的 OFT、DyLoRA、T-LoRA，
  以及（仅限显式指定的融合后端）层超出该内核支持范围的情况
  （LoHa 秩超过 `lycoris.kernels.limits.MAX_HADA_RANK`、
  LoKr 因子超过内核上限、OFT 块超过后端上限）。

每个内核算子都将 LyCORIS 融合算子（`lycoris.kernels.autograd`，通过 `backend=` 指定）
与其即时执行公式（后端 `"torch"` 上的公共 `lycoris.functional` 入口）配对。
在 train() 与 eval() 之间不同的前向（dropout、DoRA detach、DyLoRA、模块 dropout）
会在模式变化时重新绑定，因此采样和验证会使用内核算子。
对于支持接收基座加法的算子，该加法放在算子内部
（docs/internals/adapter-binding.md#forward-wrappers）。

<a id="kernelslycoris"></a>
### `KERNELS["lycoris"]`

内核算子的运行方式：

- **`auto`（默认）：** 在拥有 LyCORIS 融合后端的设备上，每个算子都是一个
  `AutotunedOp`：将设备上最高优先级后端的融合内核与即时执行公式的
  `torch.compile` 版本比较，由内核调优器对每个问题桶测量一次
  （在真实操作数的分离副本上进行前向 + 反向），按显卡保存并绑定
  （docs/internals/adapter-binding.md#autotuned-ops）。
  - 在启用梯度检查点的前向或反向内调用时，如果选择尚未确定，就运行该内核，
    从而使梯度检查点重计算重新运行其前向使用的实现。
  - 因此，`AdapterSet.prewarm(rows, dtype, devices)` 会在任何梯度检查点之外，
    使用临时输入，覆盖每个词元桶和梯度模式，为每种不同的算子（包括流式层）确定选择
    （因子保留其值且不保留梯度）；除非 `TUNE="off"`，训练构建会在第 0 步前运行它。
  - 测量比较速度；测试套件根据 PyTorch 检查精度
    （`tests/adapters/test_kernel_ops.py`）。
  - 没有融合后端时（CPU、XPU），每个算子都运行即时执行公式。
- **后端名称**（`triton` | `tilelang` | `compile` | `torch`）：每个算子都使用该后端
  （融合内核、编译后的公式、即时执行公式），不做测量；设备无法运行的后端名称会报错。

<a id="re-binding"></a>
### 重新绑定

- 基座输出始终来自 `MossLinear._impl`，基座权重来自每个 Target 一次性选择的
  来源（稠密参数、量化 / 流式存储，或覆盖值）。
- 附加、合并或移除适配器会重新绑定所属主干的算子（`bind_ops()`），
  因此融合 linear+epilogue 内核绝不会绕过适配器读取权重。
- 改变倍率（`set_multiplier`）或适配器状态（`load_state_dict`）会重新绑定每层；
  因子的数据类型和 `requires_grad` 标志属于层绑定前向的一部分，
  因此改变它们需要调用 `AdapterSet.bind()`。

<a id="steps"></a>
## 训练步骤

训练循环在每次前向之前（每个微批次，包括验证）调用 `begin_step(sigma, seq, step)`，
并在该步最后一次反向之后调用 `end_step()`：

- **T-LoRA 掩码：** 每个词元轴一个。联合块层为每个联合词元读取一行，
  文本细化器层为每个文本词元读取一行，视频嵌入器 / 最终线性层为每个图像词元读取一行，
  调制层为每个样本读取一行（`adapters.train.tlora.AXES`，根据各层路径设置）。
  `end_step` 清除它们，因此采样与导出看到的是满秩适配器，导出文件中保存的也正是它。
- **随机抽样：** 每层的 DyLoRA 秩前缀和模块 dropout 跳过决策来自一个以
  (`SEED`, step, microbatch index) 为种子的生成器，在前向之外只抽取一次：
  梯度检查点重计算会重放这些结果，每个数据并行进程抽到相同的结果，
  恢复运行也会重复这些结果。不调用 `begin_step` 时，DyLoRA 使用每个块，且不跳过任何层。

<a id="quantized-and-streamed-bases"></a>
## 量化与流式基座

当一层的基座不是普通的 16/32 位稠密参数时（通过 `MossLinear.set_base`
使用量化或流式存储，或使用 fp8 计算副本），其稠密参数是一个空占位符，
基座矩阵乘法仍使用存储自身的 `linear`。

- 这些层只绑定旁路形式：适配器增量加到基座输出上，不持有稠密权重或 dW，
  而 `ADAPTER_MODE="merge"` 会触发指明基座的验证错误。
- DoRA 每步读取基座的一个临时副本，以计算 W + dW 的范数
  （docs/internals/adapter-binding.md#quantized-and-streamed-bases）。
- `merge_into` 不接受这些层，而合并导出每次对一层的存储反量化。
