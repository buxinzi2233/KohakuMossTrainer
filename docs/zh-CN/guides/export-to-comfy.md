[English](../../guides/export-to-comfy.md)

<a id="export-to-comfyui"></a>
# 导出到 ComfyUI

训练运行结束时会把导出写入 `runs/moss/<RUN_NAME>/export/`，并在 `CHECKPOINT_EXPORT=True`
（默认）时写入每个检查点的 `export/` 文件夹；`scripts/export/export.py` 可再次导出任意
检查点（中断运行的检查点也可以）。

<a id="what-a-run-writes"></a>
## 一次运行会写出什么

| 控制项 | 取值 |
| --- | --- |
| `SAVE_FORMAT` | `comfy`（默认）、`hf`、`both`、`none` |
| `SAVE_MERGED` | `False`（默认）；`True` 额外写出合并后的 DiT（仅适配器） |
| `SAVE_FUSED_QKV` | `auto`（默认：与本次运行所训练的 DiT 文件存储 q / k / v 的方式一致；Comfy-Org 文件为融合形式）、`True`、`False` |
| `CHECKPOINT_EXPORT` | `True`（默认）：每个检查点也保存这些导出 |

| 运行 | `comfy` | `hf` |
| --- | --- | --- |
| 适配器 | `<run>-comfy.safetensors`：ComfyUI LyCORIS 文件（`lora_unet_...` 键，按 `SAVE_FUSED_QKV` 融合 q/k/v） | `<run>-lycoris.safetensors`：使用 diffusers 模块键的 LyCORIS 文件 |
| 适配器，`SAVE_MERGED` | + `<run>-merged.safetensors`：一个 ComfyUI DiT 文件（裸键，bf16） | + `<run>-diffusers/`：HF transformer 文件夹 |
| 全量微调 | `<run>-merged.safetensors`：一个 ComfyUI DiT 文件 | `<run>-diffusers/` |

- `both` 写出两列；`none` 不写任何导出（检查点仍保存训练状态，`export.py` 可读取）。
- 构建日志会列出一次导出的文件名及其大小；启用 `CHECKPOINT_EXPORT` 后每个检查点都会增加
  这些占用（`KEEP_LAST` 会随检查点一起清理它们）。
- 在 int8 DiT 上运行得到的合并文件，是反量化的基础权重加上 bf16 的 dW。
- 流水线运行的检查点导出会先把各阶段的适配器交给 rank 0（与最终导出是同一步）。
- 较早的 `EXPORT_FORMATS` 列表会在校验时映射到这些控制项（一行 INFO）：`comfy` /
  `lycoris` 是适配器列，`merged` / `diffusers` 是 DiT 列（适配器运行下的 `SAVE_MERGED`），
  空列表即 `none`。

<a id="exporting"></a>
## 导出

```bash
# ComfyUI adapter file from a checkpoint (WEIGHTS: the ComfyUI folder or the HF dir)
kogine run scripts/export/export.py --set WEIGHTS=ComfyUI/models \
    --set ADAPTER=runs/moss/<RUN_NAME>/checkpoints/step_00001000/trainable.safetensors \
    --set OUT=moss_lora.safetensors
# merged transformer (one ComfyUI DiT file)
kogine run scripts/export/export.py --set FORMAT=merged ...
# the LyCORIS file
kogine run scripts/export/export.py --set FORMAT=hf ...
# merge any adapter files (LyCORIS, ComfyUI, training states) into the base
kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
```

- `FORMAT`：`comfy`、`hf`（= `lycoris`）、`lycoris`、`merged`、`diffusers`；全量微调检查点
  （`ALGO="full"`）导出为 `merged`（`comfy`）或 `diffusers`（`hf`）。
- `FUSED_QKV`（`auto` 默认：与 `WEIGHTS` 存储 q / k / v 的方式一致）为 `comfy` 和 `merged`
  写出 Comfy-Org 文件的融合布局。
- 以 ComfyUI 文件夹作为 `WEIGHTS` 时，`TRANSFORMER` / `VARIANT` 像训练时一样挑选 DiT 文件
  （[weights.md](weights.md)）。
- 与其他导出脚本一样，`SAVE_DTYPE`（默认 `bf16`，也可为 `fp16`、`fp32`）是 `export.py` 和
  `merge.py` 写入张量的数据类型。

<a id="loading-the-file"></a>
## 加载文件

- 如果有 ComfyUI MOSS-Image 加载器，使用 LoRA 加载器节点的默认（合并）模式加载适配器。
- 如果没有，`scripts/sample/sample.py` 可直接使用 `comfy` 文件采样（`ADAPTERS`，[inference.md](inference.md#cli)）。

`comfy` 格式面向这样的 ComfyUI MOSS-Image 加载器：它将 `lora_unet_<module path
with _>.<tensor>` 或 `diffusion_model.<module path>.<tensor>` 键映射到 transformer 自身的 state-dict 名称，并将完整 transformer 作为使用无前缀键的单个 safetensors 文件加载。键布局遵循 ComfyUI 原生 MOSS-Image 模型的模块路径；融合重打包版本见[融合 q/k/v](#fused-qkv)。权重运算已对照 ComfyUI 的 `comfy/weight_adapter/*` 检查：LoRA / LoHa / LoKr 的 alpha-over-rank、DoRA 在输出轴按 `||W||` 归一化、OFT 合并忽略 `rescale`，强度在旋转内部及旋转结果之上应用。

<a id="folds-and-fp16-fix-weights"></a>
## 折叠与 fp16-fix 权重

检查点的 `trainable.safetensors` 在元数据中记录导出所需的信息：

| 键 | 内容 |
| --- | --- |
| `kohakumoss_spec` | 适配器配置 |
| `kohakumoss_fold` | 各层训练时所用的权重折叠：fp16-fix WEIGHTS 文件的折叠，或无折叠 |

- `export.py` 在读取状态之前将 WEIGHTS 转入该折叠，因此该模型的任意 WEIGHTS 文件都可使用，输出时会撤销折叠；未记录折叠的状态是在按加载状态保持不变的 WEIGHTS 上训练的。
- 对 fp16-fix WEIGHTS 文件运行 `merge.py` 时，会使用折叠的逐行缩放因子合并每个 dW，因此合并后（未折叠）的文件是精确的；再对其运行 `scripts/export/fp16_fix.py`，即可写入合并模型的 fp16-fix 检查点。

<a id="what-the-comfy-format-contains"></a>
## `comfy` 格式包含什么

每层的写入方式都使 ComfyUI 的权重适配器在强度为 1 时计算出训练得到的 dW：

| 算法 | 张量 | 转换 |
| --- | --- | --- |
| LoRA / LoCon | `lora_up.weight`、`lora_down.weight`、`alpha` | 无（标量折叠进 up） |
| LoHa | `hada_w{1,2}_{a,b}`、`alpha` | 无 |
| LoKr | `lokr_w1` / `lokr_w1_{a,b}`、`lokr_w2` / `lokr_w2_{a,b}`、`alpha` | 无 |
| GLoRA | `{a1,a2,b1,b2}.weight`、`alpha` | 无 |
| DoRA（输出轴） | + `dora_scale` [out, 1] | `dora_scale * \|\|W\|\| / \|\|W + dW\|\|`：ComfyUI 按该轴上未合并的权重归一化 |
| DoRA（输入轴） | + `dora_scale` [1, in] | 无 |
| OFT | `oft_blocks`、`alpha` | `alpha` 是有效约束（LyCORIS 存储的是 constraint / out_dim） |
| 带 `rescaled` 的 OFT | `diff` | ComfyUI 的合并忽略 `rescale`，因此写入完整 dW |
| BOFT | `oft_blocks`、`rescale`、`alpha` | `alpha` 与 OFT 相同 |
| IA3 | `diff` | 完整 dW，由 ComfyUI 的 `diff` 适配器加载 |
| DyLoRA | `lora_up.weight`、`lora_down.weight`、`alpha = rank` | 所有 rank，求值缩放折叠进 up |
| T-LoRA | rank-2r LoRA | `[P*lam*s, -P0*lam0*s] @ [Q ; Q0]`，无 rank 掩码 |
| full | `diff`、`diff_b` | 无 |
| 归一化层 | `w_norm`、`b_norm` | 无 |

<a id="fused-qkv"></a>
## 融合 q/k/v

联合块的 q/k/v 在 KohakuMoss 中是独立的层，写入为独立的 `..._proj_q` / `_k` / `_v` 键。在 ComfyUI 中：

- MOSS-Image 模型（`comfy/ldm/moss_image`）在未融合检查点中将它们命名为 `layers.N.proj.{q,k,v}`，在 Comfy-Org 重打包版本中则为 `layers.N.proj.qkv`（按行 q;k;v，`fused_qkv=True`）；
- 通用 LoRA 键映射（`comfy/lora.py`、`model_lora_keys_unet`）将 `lora_unet_<path with _>` 和 `diffusion_model.<path>` 映射到模型自身的 `<path>.weight` 键。

因此，未融合的文件面向未融合检查点，而融合重打包版本使用融合文件。`SAVE_FUSED_QKV` /
`FUSED_QKV` 为 `auto` 时跟随本次运行所训练的 DiT 文件（Comfy-Org 文件：融合；HF
transformer 文件夹：未融合）；`True` / `False` 则强制其一。

融合的 `comfy` 文件会把每个块的 q/k/v 写成一个 `layers.N.proj.qkv` 层：

| q/k/v 层 | 写入形式 |
| --- | --- |
| 三组普通 LoRA | 一个 rank 为 r_q + r_k + r_v 的 LoRA：down 为三个 down 堆叠，up 为分块对角矩阵并折叠进各层的 alpha / rank，alpha = rank |
| 其他任何类型（DoRA、LoHa、LoKr、OFT 及其余类型） | 精确的 `diff`：将三个 dW 按行堆叠；`diff` 随强度线性缩放 |

KohakuMoss 自身的加载器会将融合 LoRA 或 `diff` 层重新拆成 q / k / v（也包括其他训练器生成的融合 LoRA）；其他融合类型会报错。

<a id="strength"></a>
## 强度

- 对于低秩、DoRA、`diff` 和归一化层格式，ComfyUI 随强度线性缩放 dW；除以 `diff` 写入的带 rescale 的 OFT 外，这与 KohakuMoss 的乘数一致。
- ComfyUI 的 OFT 和 BOFT 合并会在旋转内部以及 dW 上再次应用强度，因此这些类型（以及带 rescale 的 OFT）在强度为 1 时精确一致。
- KohakuMoss 自身的加载器（`kohakumoss.adapters.load_adapters`）会为所有类型复现训练时的乘数。
