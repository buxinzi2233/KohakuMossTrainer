[English](../../internals/adapter-kinds.md)

<a id="adapter-kind-records"></a>
# 适配器类型记录

`kohakumoss.adapters.kinds.Kind` 保存应用器针对各 LyCORIS 算法所采取的不同行为：

- Target 在构建时根据模块类查找一次对应记录；
- 绑定器（[adapter-binding.md](adapter-binding.md)）、导出器和推理栈读取记录中的字段；
- `adapters.register(kind)` 添加或替换一条记录。

<a id="fields"></a>
## 字段

| 字段 | 含义 |
| --- | --- |
| `name`, `module`, `algos` | 类型名称、LyCORIS 模块类、`ALGO` 名称 -> LyCORIS `network_module` |
| `diff_weight(t)` | 乘数为 1 时的 dW（fp32，DoRA 之前）；由 `Target.unit_fn` 绑定 |
| `diff_weight_at(t, m)` | 当结果不是 `diff_weight * m` 时，在 `m` 处构建 (dW, db) 的构建器（OFT / BOFT 旋转），或 None |
| `kernel_diff_weight(t, ops)` | 由层的内核算子在乘数为 1 时实现 W -> dW，或 None |
| `kernel_merged_weight(t, ops)` | 在乘数为 1 时实现 W -> W + dW，加法在算子内部进行（`base=`），或 None |
| `kernel_bypass(t, ctx)` / `kernel_merge(t, ctx)` | 由内核算子针对该上下文构建的绑定前向函数；模块路径则为 None |
| `module_forward(t, ctx)` | 基于模块方法的旁路前向函数（默认：模块的 `bypass_forward_diff`） |
| `own_forward(t, ctx)` | 完全替换选择逻辑（DyLoRA 的训练 / 评估旁路前向函数） |
| `comfy_tensors(t)` | ComfyUI 张量集合（默认：将 DoRA 幅值折叠后的原生集合） |
| `undo_comfy(lyco, parts)` | 加载时撤销 KohakuMoss 自身的 ComfyUI 转换 |
| `uniform_scale_tensors(tensors)` | 均匀行缩放所乘的张量（每一项取一个因子） |
| `row_scale_tensors(tensors)` | 逐行缩放所乘的输出行所在的张量 |
| `stacking` | 推理堆叠方式：`additive`（旁路中的逐行缩放链）、`commuting`（块均匀缩放可交换）、`chained`（旁路链，折叠缩放下为稠密形式）、`dense` |
| `bypass_exact_at(m)` | 模块的旁路在 `m` 处是否精确（OFT：仅在 1 处） |
| `loraplus_tensors` | LoRA+ 提升的零初始化因子 |
| `per_token_mask` | 逐 token 秩掩码（T-LoRA）：仅支持旁路，每个轴一组掩码 |
| `draw_step(t, gen)` | 每步随机抽取（DyLoRA 的秩前缀） |
| `supports_dora`, `rankless`, `training_dependent` | 允许 DoRA、无需 `DIM`、在训练 / 评估切换时重新绑定前向函数 |

<a id="scaling-dw"></a>
## 缩放 dW

对 dW 进行均匀行缩放 c 时，通过乘上每个加法项的一个因子来写入：

| 类型 | 因子 |
| --- | --- |
| LoRA | `lora_up` |
| LoHa | `hada_w1_a` |
| LoKr | `lokr_w1`（或 `lokr_w1_a`） |
| T-LoRA | `lambda_layer` 和 `base_lambda` |
| GLoRA | 仅 `b1`（其 W*a1*a2 项随基座一起缩放） |

旋转和 IA3 与其基座的均匀缩放可交换。逐行缩放或没有这类因子的集合则写为精确的 `diff`。

<a id="presets"></a>
## 预设

LyCORIS 将预设保存在 `LycorisNetwork` 的类状态中，因此 `resolve_preset` 每次调用都会设置所有预设键。
