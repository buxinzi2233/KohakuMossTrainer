[English](../../guides/full-finetune.md)

<a id="full-finetune"></a>
# 全量微调

`ALGO="full"` 训练骨干网络自身的参数（除非设置 `TRAIN_TEXT_REFINER=True`，否则文本精炼器保持冻结）。参数通过 `FullTrainable` 原位训练。

`configs/moss/_base/base_full.py` 遵循 MOSS 技术报告：

- Muon + AdamW，学习率 5e-5；
- 权重衰减 0.01，偏置 / 归一化层 / 嵌入器不使用权重衰减；
- 对最后一个块的 q/k 归一化权重使用趋向 1 的衰减；
- 梯度裁剪 1.0，EMA 0.9999。

<a id="choosing-a-setup"></a>
## 选择方案

每个参数的状态约为 16 字节（fp32 主副本、两个 Adam 矩估计、梯度），因此 5.1 B 参数的骨干网络需要以下方案之一：

| 方案 | 配方 |
| --- | --- |
| 4x 32 GB，使用 FSDP2（Linux） | `recipes_full_fsdp_4xv100.py` |
| 一个 24 GB GPU，主机主副本 + 权重流式传输 | `recipes_full_24g_offload.py` |
| 一个 RTX 5090 32 GB，主机主副本 + 权重流式传输，每步 36864 个 token | `recipes_full_5090.py` |
| 多个 24 GB GPU（Linux），在相同的流式传输配置档上使用 DDP | `kohakumoss-config new train --algo full --gpus N` |
| PCIe 上的 4x RTX 5090 32 GB（Linux），在 5090 流式传输配置档上使用 DDP | `recipes_full_ddp_4x5090.py` |

全量微调训练 16 位 DiT：在 ComfyUI 文件夹中它使用
`diffusion_models/moss_image_<VARIANT>_bf16.safetensors`（`ALGO="full"` 的默认值），
在 HF 布局中则是 `transformer/` 文件夹。

```bash
# a config from the presets for this GPU, or one that extends a recipe
kohakumoss-config new train --algo full --weights ComfyUI/models --data <dataset> --name my-ft
kohakumoss-config new --from full_24g_offload my_ft.py    # then set WEIGHTS / DATASETS in it

# the data passes printed by the first command, then
kogine run scripts/train/moss.py --config configs/user/my-ft.py
```

<a id="one-gpu"></a>
## 单 GPU

`kohakumoss-config new train --algo full` 选择显存层（[configs.md](configs.md#layers)）：

| 显卡 | 层 |
| --- | --- |
| 在 `--vram-tolerance` 容差内可容纳 24 GiB（默认 0.05：从 22.8 GiB 起，因此报告为 23.99 GiB 的 24 GB 显卡会选中它） | `24g_full` |
| 超过 24 GiB | `auto` |
| 更少 | 拒绝 |

单 GPU 流式传输配方：

- 使用 AdamW 训练（Muon 在反向传播后对整个矩阵执行更新，而换出的块不允许这样做），关闭 EMA；
- 需要约 70 GiB 主机 RAM（[主机 RAM](#host-ram)）。

<a id="several-gpus-linux"></a>
## 多 GPU（Linux）

使用 `--gpus N` 时，起始配置按以下规则选择：

| 条件 | 引擎 |
| --- | --- |
| 运行 `kohakumoss-config` 的机器上 N 张显卡的每一对都具有对等访问（NVLink / P2P），且每个 rank 的 FSDP2 占用（`fsdp` 层的常驻配置档，每个参数、梯度和优化器状态均分片到 N 个 rank；按内存估计）在 `--vram-tolerance` 容差内能放入显卡 | FSDP2 |
| 否则 | DDP：全量微调显存层的换入换出配置档（`ckpt-ring4-actoff-optstream-gradoff`）在每个 rank 上原样运行，各 rank 共享一份主副本的主机拷贝（[multi-gpu.md](multi-gpu.md#shared-host-masters)） |
| `--vram auto` | DDP |

因此 NVLink 上的 4x 32 GB V100 使用 FSDP2，而普通 PCIe 上的显卡使用 DDP：在 4x RTX 5090
（PCIe，无 P2P）上，全量微调在 DDP 下达到 15.3k tok/s，而 FSDP2 下为 8.3k
（[training-5090.md](../performance/training-5090.md)）。两个 RTX 3090 在该配置档下使用 DDP
测得 0.356 samples/s（扩展效率 0.79），每张显卡的 nvidia-smi 峰值为 17.3 GiB，总主机 RAM 为 63 GiB。

<a id="fp8--mxfp8-compute-sm89"></a>
## fp8 / MXFP8 计算（sm89+）

`BASE_PRECISION="mxfp8"`（或 `"fp8_compute"` 配合 `FP8_RECIPE`）使用每次优化器更新后重新量化的权重副本，以 fp8 运行每个块内线性层的前向计算和输入梯度计算。

- 权重梯度和优化器保持 16-bit / fp32。
- 权重保持常驻（`stream="off"`，例如 `MEMORY_PROFILE="ckpt-resident"`），每个权重额外占用 1（`FP8_DGRAD="16bit"`）或 2 字节。

见 [quant-kernels.md](../internals/quant-kernels.md#full-finetune)。

<a id="exports-and-ema"></a>
## 导出与 EMA

| `SAVE_FORMAT` | 输出 |
| --- | --- |
| `"comfy"`（默认） | `<RUN_NAME>-merged.safetensors`：一个使用无前缀键的 safetensors 文件（[export-to-comfy.md](export-to-comfy.md) 中的 ComfyUI 单文件布局；按 `SAVE_FUSED_QKV` 融合 q / k / v） |
| `"hf"` | `<RUN_NAME>-diffusers/`：分片的 transformer 文件夹 |
| `"both"` / `"none"` | 两者 / 都不 |

- 在 ComfyUI 文件夹中，全量微调默认使用 bf16 DiT 文件；int8 文件仅在
  `BASE_PRECISION="compute"` 下训练（使用其反量化权重，[weights.md](weights.md#dit)）。

- 启用 `EMA` 时，每个检查点都会包含平均权重（`ema.safetensors`）；使用幂次 EMA 时，`kohakumoss.training.ema.posthoc` 可在训练后重建任意 EMA 宽度（[training.md](../concepts/training.md#ema)）。
- 使用 `scripts/sample/sample.py --set
  TRANSFORMER=<run>/export/<RUN_NAME>-merged.safetensors` 对合并文件采样（文本编码器和 VAE 来自 `WEIGHTS`；[inference.md](inference.md#cli)）。

<a id="host-ram"></a>
## 主机 RAM

单 GPU 配方（`ckpt-ring4-actoff-optstream-gradoff`，换入换出的块）在主机上保留：

| 部分 | 大小 |
| --- | --- |
| 每个块参数：fp32 主副本、两个 fp32 Adam 矩估计和 16-bit 流式传输副本（4.76 B 个块参数中的每个参数占 14 字节） | 约 62 GiB |
| 当 `GRAD_ACC > 1` 时：每个参数一份 fp32 主机梯度累加和（4 字节），在首个发送该梯度的微批次时分配 | 额外约 18 GiB |
| 训练器进程 | 预算 8 GiB（基准矩阵的预算，并非测量值） |
| 8192 个 token 时卸载的激活值 | 约 2 GiB |
| 构建时：用于采样提示词的文本编码器（[configs.md](configs.md#host-ram)） | 15.63 GiB |

请为训练步骤期间预留约 70 GiB，构建时预留 88 GiB。上面各 rank 共享一份副本的双 GPU DDP 运行测得总主机 RAM 为 63 GiB。在 FSDP2 下，主副本和状态会分片到各 GPU 上。
