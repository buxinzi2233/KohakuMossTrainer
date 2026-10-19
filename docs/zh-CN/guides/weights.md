[English](../../guides/weights.md)

<a id="weights-where-each-file-goes"></a>
# 权重：每个文件放在哪里

MOSS-Image 提供两种布局，KohakuMoss 在训练、采样、缓存流程和导出脚本中都能读取：

- ComfyUI 布局：ComfyUI `models/` 文件夹下每个组件一个文件，即 Comfy-Org/MOSS-Image
  重打包（随附配置中的默认布局）；
- HF 布局：OpenMOSS-Team/MOSS-Image-Base / -SFT，一个含 `transformer/`、`text_encoder/`、
  `tokenizer/`、`vae/` 的流水线文件夹。

<a id="comfyui-layout"></a>
## ComfyUI 布局

```
models/
  diffusion_models/  moss_image_{base,sft}_int8_convrot.safetensors   (5.4 GB)
                     moss_image_{base,sft}_bf16.safetensors           (10.2 GB)
  text_encoders/     qwen3.5_9b_int8_convrot.safetensors              (9.7 GB)
                     qwen3.5_9b_bf16.safetensors                      (19.3 GB)
                     qwen3.5_9b_w4a8_gptq.safetensors                 (7.3 GB)
  vae/               flux2-vae_small.safetensors                      (0.25 GB)
                     flux2-vae.safetensors                            (0.34 GB)
```

```bash
hf download Comfy-Org/MOSS-Image diffusion_models/moss_image_base_int8_convrot.safetensors \
    text_encoders/qwen3.5_9b_int8_convrot.safetensors vae/flux2-vae_small.safetensors \
    --local-dir ComfyUI/models
```

`WEIGHTS` 就是 `models/` 文件夹。每个组件都有一个配置项用于指定其文件（该组件子文件夹下的
文件名，可带可不带 `.safetensors`，也可以是任意路径）；留空时取第一个存在的默认文件：

| 配置项 | 子文件夹 | 默认值，按顺序 |
| --- | --- | --- |
| `TRANSFORMER` | `diffusion_models/` | `moss_image_<VARIANT>_int8_convrot`、`moss_image_<VARIANT>_bf16`（全量微调：优先 bf16 文件） |
| `TEXT_ENCODER` | `text_encoders/` | `qwen3.5_9b_int8_convrot`、`qwen3.5_9b_bf16`、`qwen3.5_9b_w4a8_gptq` |
| `VAE` | `vae/` | `flux2-vae_small`、`flux2-vae` |

- 训练脚本（`scripts/train/moss.py`）、采样与网格脚本（`scripts/sample/`）以及导出脚本
  （`scripts/export/export.py`、`merge.py`：`TRANSFORMER`、`VARIANT`）接受同样的配置项。
- 缓存流程把该文件夹作为 `TEXT_ENCODER_KWARGS["path"]` / `IMAGE_ENCODER_KWARGS["path"]`
  （默认的文本编码器和 VAE 文件），也可以直接给一个文件。
- `configs/moss/weights/weights_comfy.py` 逐一指定每个文件；`weights_hf.py` 对应 HF 布局。
- 文件缺失属于配置项问题，报错会列出尝试过的每一个路径。

<a id="dit"></a>
### DiT

两个 DiT 文件以裸键保存 MOSS-Image 的 transformer，联合块的 q / k / v 已融合
（`layers.N.proj.qkv`，行 q;k;v）。KohakuMoss 在加载时把它们拆分为自己的 `q` / `k` / `v`
层。

int8 文件按[量化文件](#quantized-files)的方式存储每个块的四个投影（`proj.qkv`、`proj.o`、
`proj.ffn_up_gate`、`proj.ffn_down`）：旋转后权重的 int8 码，每行一个 fp32 缩放，
其余全部为 bf16。

- `BASE_PRECISION="auto"`（默认）保留这些 int8 基础权重：冻结的基座运行旋转后的 int8 线性层
  （`docs/internals/quant-kernels.md#rotated-int8-linear`），适配器以旁路方式挂载，
  永不会构建 16 位权重。`auto` 下的 bf16 文件按 16 位运行（`compute`）。
- 其他任何 `BASE_PRECISION` 都会先反量化 int8 层（W = s q R，一行 INFO 日志）：
  `compute` 以 16 位运行反量化后的权重，`int8` / `fp8` / `fp8_compute` / `mxfp8`
  则像对待 16 位文件那样对它们（以及其他线性层）做量化。
- 全量微调训练的是稠密权重：对 int8 文件需要 `BASE_PRECISION="compute"`（训练反量化后的
  权重）；`auto` 会被配置项校验拒绝。全量微调的默认值会选择 bf16 文件。
- 每个块的 `comfy_attention.config` 张量是 ComfyUI 的运行时提示（其注意力内核）；
  KohakuMoss 不读取它（会有一行 INFO 日志统计它们）。
- 内存规划器在 `auto` 下按 int8 字节计算。

<a id="text-encoder"></a>
### 文本编码器

ComfyUI 的文本编码器文件就是完整的 Qwen3.5-9B 检查点：语言模型在 `model.language_model.*`
下，视觉塔（`model.visual.*`）、MTP 头（`mtp.*`）和 LM 头（`lm_head.*`）也都在里面。
MOSS-Image 读取语言模型的最后一层隐藏状态，因此 KohakuMoss 只把 `model.language_model.*`
加载到文本模型自己的键上，其余三部分不读取。

| 文件 | 语言模型线性层 | 嵌入 |
| --- | --- | --- |
| `qwen3.5_9b_bf16` | bf16 | bf16 |
| `qwen3.5_9b_int8_convrot` | int8 ConvRot（组 256） | int8 ConvRot |
| `qwen3.5_9b_w4a8_gptq` | W4A8：4 位码，组 16，ConvRot 256 | int8 ConvRot |

- `TEXT_ENCODE["precision"]="auto"`（默认）保留文件存储的形式：int8 线性层和嵌入按存储形式
  运行，W4A8 线性层把其码解码到同一个 int8 网格。在主机上（memory 为 `cpu`）`auto`
  即 `compute`。
- `TEXT_ENCODE["w4a8"]` 选择 W4A8 解码方式：`int8` 在加载时解码一次（每权重 1 字节，
  每次调用使用 int8 线性层），`packed` 保留 4 位码（每权重 0.56 字节）并在每次调用时解码，
  `auto`（默认）在解码后的编码器能与 DiT 步骤一同常驻时选 `int8`，否则选 `packed`。
  两者给出相同的 int8 数值；开销见 `docs/performance/quant-rotate.md#w4a8-decode`。
- 其他任何精度都会先把存储的层反量化成 16 位模型，再像对待 HF 检查点那样量化。
- 文本缓存键记录存储布局（`src:int8-convrot256`，……），因此只有文件格式相同时缓存才会被
  复用。

ComfyUI 文件不带配置，也不带分词器。KohakuMoss 把两者都放在
`kohakumoss/encoders/text/assets/qwen3.5-9b/`（来自 MOSS-Image 的 `text_encoder/` 和
`tokenizer/` 文件夹）：`config.json`、`tokenizer.json.gz`（3.8 MB，加载时在内存中解压）、
`tokenizer_config.json`、`chat_template.jinja`。打包的分词器给出与 HF `tokenizer/`
文件夹相同的 token id 和聊天模板。

<a id="vae"></a>
### VAE

`VAE`（以及潜变量缓存的 `IMAGE_ENCODER_KWARGS["path"]`）接受两个文件中的任意一个：

| 文件 | 键 | 解码器 | 潜变量 |
| --- | --- | --- | --- |
| `flux2-vae_small.safetensors` | diffusers（`encoder.down_blocks.N.resnets.N...`） | 宽度 96 / 192 / 384 / 384，与 HF `vae/` 解码器完全一致 | fp32 编码器，其 bf16 舍入结果由 HF `vae/` 文件夹保存 |
| `flux2-vae.safetensors` | BFL / ldm（`encoder.down.N.block.N...`） | 宽度 128 / 256 / 512 / 512（完整的 FLUX.2 解码器） | 与小文件完全相同的编码器：同样的潜变量 |

- 小文件是 fp32 的 MOSS-Image VAE。其解码器与 HF `vae/` 解码器逐张量一致。其编码器、
  `quant_conv` 和潜变量统计量（`bn.*`）都是 fp32；HF 文件夹保存的是舍入到 bf16 的同样
  张量。两者给出的潜变量差异就在该舍入的层面上（一张 256 x 256 数据集图像上，fp32 的相对
  L2 差异为 4.6e-3；从文件潜变量解码的 PSNR 为 37.08 dB，从文件夹潜变量解码为
  37.09 dB），因此潜变量缓存由哪个 VAE 编码，就得由同一个 VAE 使用。
- 完整文件包含小文件的编码器（同样的潜变量、同样的潜变量缓存）和 FLUX.2 原有的更宽解码器
  （同一张图像上解码 PSNR 为 37.07 dB）。
- 单个文件以 fp32 加载（归一化统计量保持 fp32），再转换为 `VAE_DTYPE`，文件夹也一样。
- BFL 的键会映射到 diffusers 的 `AutoencoderKLFlux2` 名称：编码器下采样块按顺序，解码器
  上采样块按逆序（BFL 的层级 0 最细，diffusers 的块 0 最粗），`nin_shortcut` ->
  `conv_shortcut`，中间注意力的 1x1 卷积 -> 线性层，`encoder.quant_conv` /
  `decoder.post_quant_conv` -> 顶层的 `quant_conv` / `post_quant_conv`，`bn.*` 不变。
- 文件的 `AutoencoderKLFlux2` 配置从其张量形状读取（块宽度、每块的层数、通道数、量化卷积、
  中间注意力）；其余字段为 FLUX.2 默认值。

<a id="hf-layout"></a>
## HF 布局

```
models/MOSS-Image-Base/   transformer/  text_encoder/  tokenizer/  vae/
```

```bash
hf download OpenMOSS-Team/MOSS-Image-Base --local-dir models/MOSS-Image-Base
```

`WEIGHTS` 是流水线文件夹（训练脚本）或一个含 `base/` 和 `sft/` 的权重根目录（采样脚本，
`VARIANT` 选择其中一个）。`TRANSFORMER` / `TEXT_ENCODER` / `VAE` 留空时分别读取
`transformer/`、`text_encoder/`（加 `tokenizer/`）、`vae/`；设置时它们是路径（单个 DiT
文件、fp16 修正检查点、另一个文本编码器文件夹或一个 ComfyUI 文件）。所有权重都是 bf16：
`BASE_PRECISION` 和 `TEXT_ENCODE["precision"]` 的 `auto` 都解析为 `compute`。

<a id="quantized-files"></a>
## 量化文件

ComfyUI 的量化层 `<n>` 把其码存为 `<n>.weight`、把缩放单独存，并用 `<n>.comfy_quant` 记录
其格式（一个 uint8 JSON）：

| 格式 | 张量 | 权重 |
| --- | --- | --- |
| `int8_tensorwise` | `weight` I8 [N, K]、`weight_scale` F32 [N, 1] | 每行 q x s |
| `float8_e4m3fn` | `weight` F8 [N, K]、`weight_scale` / `scale_weight` | q x s |
| `asym_w4a8_int8` | `weight` I8 [N, K/2]（每字节两个 4 位码）、`weight_s_rel` F8 [N, K/16]、`weight_s_channel` F32 [N]、`weight_codebook` F32 [16] | round(codebook[code] x s_rel)（int8 网格）x s_channel |

当 `"convrot": true` 时，存储的权重是旋转后的权重 W R，其中 R = blockdiag(H_g) 在 K 上
分块对角，H_g 是 `convrot_groupsize` g 的归一化正则 Hadamard 矩阵（ConvRot）；加载器把
旋转与基础权重一同保留，每个使用方读取 W（线性层旋转其输入，反量化后的权重会被旋转回来）。
没有记录文件时，int8 / fp8 的 `<n>.weight` 与 `weight_scale` / `scale_weight` 相邻，
会以同样方式读取，但不带旋转。这些公式遵循 comfy-kitchen 的 eager 后端；
`tests/quant/test_comfy_formats.py` 用独立的 numpy 移植逐一校验。

<a id="saving"></a>
## 保存

导出默认按 ComfyUI 布局写出（`SAVE_FORMAT`、`SAVE_MERGED`、`SAVE_FUSED_QKV`）：
[export-to-comfy.md](export-to-comfy.md)。

<a id="third-party-files"></a>
## 第三方文件

打包的 Qwen3.5-9B 配置和分词器来自 OpenMOSS-Team/MOSS-Image-Base（Qwen 团队的
Qwen3.5-9B 分词器，Qwen/Qwen3.5-9B 和 Comfy-Org/MOSS-Image 也随附该分词器），全部遵循
Apache License, Version 2.0。许可证文本和列出各来源的 NOTICE 就在这些文件旁边
（`kohakumoss/encoders/text/assets/qwen3.5-9b/LICENSE`、`NOTICE`）；wheel 会一并分发。
