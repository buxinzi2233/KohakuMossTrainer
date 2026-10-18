[English](../../guides/export-to-comfy.md)

<a id="export-to-comfyui"></a>
# 导出到 ComfyUI

训练运行结束时，会将 `EXPORT_FORMATS` 指定的格式（默认 `comfy`；随附配置还添加了 `lycoris`）写入 `runs/moss/<RUN_NAME>/export/`。`scripts/export/export.py` 可导出某次运行的任意检查点（中断运行的检查点也可以）。

<a id="exporting"></a>
## 导出

```bash
# adapter file from a checkpoint
kogine run scripts/export/export.py --set WEIGHTS=<model dir> \
    --set ADAPTER=runs/moss/<RUN_NAME>/checkpoints/step_00001000/trainable.safetensors \
    --set OUT=moss_lora.safetensors
# merged transformer (fused q/k/v like the Comfy-Org repack)
kogine run scripts/export/export.py --set FORMAT=merged --set FUSED_QKV=True ...
# adapter file for the fused-q/k/v Comfy-Org repack (see "Fused q/k/v")
kogine run scripts/export/export.py --set FUSED_QKV=True ...
# merge any adapter files (LyCORIS, ComfyUI, training states) into the base
kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
```

- 全量微调检查点（`ALGO="full"`）导出为 `merged` 或 `diffusers`。
- 与其他导出脚本一样，`SAVE_DTYPE`（默认 `bf16`，也可为 `fp16`、`fp32`）是 `export.py` 和 `merge.py` 写入张量的数据类型。

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

因此，默认文件面向未融合检查点；融合重打包版本使用下面的 `FUSED_QKV=True` 文件。

`FUSED_QKV=True` 配合 `FORMAT="comfy"`，会将每个块的 q/k/v 写成一个 `layers.N.proj.qkv` 层，供融合重打包版本使用：

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
