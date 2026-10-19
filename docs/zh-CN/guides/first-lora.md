[English](../../guides/first-lora.md)

<a id="your-first-lora"></a>
# 你的第一个 LoRA

从安装到使用适配器采样，共七步。以下步骤在克隆仓库的根目录执行，适用于 Linux 或 Windows（单 GPU）。

Windows 上先激活 venv，再在 PowerShell 或 `cmd` 中使用相同命令：

- PowerShell：`.venv\Scripts\Activate.ps1`（每个用户先执行一次 `Set-ExecutionPolicy -Scope
  CurrentUser RemoteSigned`；默认策略会拒绝脚本）；
- `cmd`：`.venv\Scripts\activate.bat`。

如果想通过少量选项而非配置文件进行一次简短运行，示例脚本会替你写入并检查配置：`scripts\examples\train.ps1`（Windows）或 `scripts/examples/train.sh`（Linux），见 [scripts/examples/README.md](../../../scripts/examples/README.md)。

<a id="steps"></a>
## 步骤

1. **安装**（[install.md](install.md)；Windows 工具链：[install-windows.md](install-windows.md)），然后检查环境：`kohakumoss-doctor`。它会列出每个 GPU 及其架构、已安装的工具链、各操作使用的内核后端，以及所有缺失内容和对应的修复命令。
2. **下载权重**到 ComfyUI models 文件夹（约 15 GB；已安装的 ComfyUI 的 `models/` 可直接使用）：

   ```bash
   hf download Comfy-Org/MOSS-Image diffusion_models/moss_image_base_int8_convrot.safetensors \
       text_encoders/qwen3.5_9b_int8_convrot.safetensors vae/flux2-vae_small.safetensors \
       --local-dir ComfyUI/models
   ```

   运行会按存储状态在 int8 DiT 和文本编码器上训练。HF 布局用法相同
   （`hf download OpenMOSS-Team/MOSS-Image-Base --local-dir models/MOSS-Image-Base`，
   约 26 GB bf16，`--weights models/MOSS-Image-Base`）；每个文件和配置项见
   [weights.md](weights.md)。
3. 为你的 GPU 和数据**编写配置**：

   ```bash
   kohakumoss-config new train --weights ComfyUI/models --data <dataset> --name my-lora
   ```

   它会写入 `configs/user/my-lora.py` 和 `configs/user/my-lora_data.py`；在第一个文件中设置 `STEPS`、`SAMPLE_PROMPTS`（默认：数据集的第一条描述文本，或 `--prompt`）以及其他设置。选项（[configs.md](configs.md)）：

   | 标志 | 值 |
   | --- | --- |
   | `--algo` | `loha`、`lokr`、`oft`、`dora`、`full` |
   | `--source` | 其他布局使用 `folder`、`folder_nl`、`jsonl`、`parquet`、`webdataset`、`moss_shard` |
   | `--vram`、`--vram-tolerance`、`--precision`、`--gpus` | 覆盖从 GPU 读取的值 |

4. **检查数据规划，然后构建文本和潜变量缓存**，只需执行一次（命令在配置的文档字符串中；[datasets.md](datasets.md) 解释输出内容）：

   ```bash
   kogine run scripts/data/inspect_data.py --config configs/user/my-lora_data.py
   kogine run scripts/data/cache_text.py --config configs/user/my-lora_data.py
   kogine run scripts/data/cache_latents.py --config configs/user/my-lora_data.py
   ```

5. **训练**：

   ```bash
   kogine run scripts/train/moss.py --config configs/user/my-lora.py
   ```

   - 任意配置项都可在命令行覆盖（`--set LR=2e-4 --set DIM=16`）。
   - 使用 `--set RESUME=latest` 恢复中断的运行。
   - 显卡上的首次运行会在第 0 步之前调优内核并保存决策；后续运行会复用这些决策（[预期表现](#what-to-expect)）。

6. 在 `runs/moss/<RUN_NAME>/` 下**查找输出**：

   | 路径 | 内容 |
   | --- | --- |
   | `config.json` | 配置项 |
   | `log.txt` | 每行控制台日志：规划、内核决策、警告、错误 |
   | `metrics.jsonl` | 每步指标 |
   | `samples/step_XXXXXXXX/` | 训练期间的采样结果 |
   | `checkpoints/step_XXXXXXXX/` | 检查点，每个都带有其 `export/`（`CHECKPOINT_EXPORT`） |
   | `export/<RUN_NAME>-comfy.safetensors`、`export/<RUN_NAME>-lycoris.safetensors` | 适配器（`SAVE_FORMAT`：基础配置写入 `both`），在运行结束时写入 |

   中断运行的检查点可通过 `scripts/export/export.py` 导出（[export-to-comfy.md](export-to-comfy.md)）。`comfy` 文件使用 ComfyUI 的 MOSS-Image 模型的键布局（可在那里用 LoRA 加载器节点加载）；第 7 步使用本仓库的流水线对它采样。

7. 使用导出的适配器**采样**（[inference.md](inference.md)）：

   ```bash
   kohakumoss-config new sample --weights ComfyUI/models --adapter runs/moss/my-lora/export/my-lora-comfy.safetensors --name my-lora-sample
   kogine run scripts/sample/sample.py --config configs/user/my-lora-sample.py --set PROMPT="a lighthouse on a rocky coast at sunset"
   ```

<a id="what-to-expect"></a>
## 预期表现

- **显卡上的首次运行**：第 0 步之前的内核预热会为该运行规划的每个内核问题计时，并保存决策；其日志行为 `prewarm: N problems from the
  tuning table, M tuned in X s`。之后在同一显卡上使用相同形状运行时，会读取表，不再计时（`M` = 0）。内核与编译路径之间的重新调优会在前 `RETUNE_STEPS` 步（默认 3）的每一步后运行，此后锁定；第一个采样网格的第一个采样器步骤会调优自身的形状（[调优窗口](../internals/tuning.md#tuning-windows)）。
- **调优配置项**：`kohakumoss-tune` 在运行之前填充表（[logging.md](logging.md#kernel-tuning)）；`--set TUNE=off` 跳过所有计时（使用已存储决策或固定默认值），用于快速冒烟运行。
- **各显卡的速度和显存**：训练进度条显示 s/it 和 `vram`（已分配峰值 / 已预留峰值 / 设备占用，[logging.md](logging.md#vram)）。各显卡上每个显存层的单步耗时和 nvidia-smi 峰值来自真实 GPU 基准矩阵的速度与容量限制运行，见其 `report.md`（[bench-matrix.md](../performance/bench-matrix.md#report)）；发布测量结果位于仓库 README 中。
- **主机 RAM**：[configs.md](configs.md#host-ram)。

<a id="recipes"></a>
## 配方

固定配方针对常见目标组合相同的层（`configs/moss/recipes/`）：

| GPU | 配方 |
| --- | --- |
| 支持 bf16 的 24 GB 显卡（3090、4090、50xx） | `recipes_lora_24g_bf16.py` |
| RTX 5090 32 GB，最高速度 | `recipes_lora_5090.py` |
| 仅支持 fp16（V100、2080 Ti） | `recipes_lora_16g_fp16.py` |
| 12 GB | `recipes_lora_12g_stream.py` |
| Intel Arc B 系列 | `recipes_lora_xpu_b50.py` |
| 两个 GPU（Linux） | `recipes_lora_ddp.py` |
| 四个 RTX 5090（Linux） | `recipes_lora_ddp_4x5090.py` |

- 24 GB bf16 显卡上的其他算法：`recipes_lycoris_loha.py`、`recipes_lycoris_lokr.py`、`recipes_lycoris_oft.py`、`recipes_dora.py`（每个都组合 `vram_24g` 和 `precision_bf16`）。
- 对于其他任何显卡或精度，使用 `kohakumoss-config new train --algo loha | lokr |
  oft | dora`。
- 设置好 `WEIGHTS` 和 `DATASETS` 后（在通过 `use_config` 引用它的文件中，或用 `--set`），通过 `kogine run scripts/train/moss.py --config
  configs/moss/recipes/<recipe>.py` 运行配方。

<a id="small-gpus"></a>
## 小显存 GPU

`kohakumoss-config new train --vram 8`（或 `--vram 6`）组合 8 GB / 6 GB 显存层（[configs.md](configs.md#layers)）：

- 权重流式传输，两个常驻块，激活值和梯度卸载，主机 AdamW 状态；
- 每批次 4352 个 token；
- 除每 `SAVE_EVERY` 步保存外，还每 30 分钟保存一个检查点（`SAVE_MINUTES`）；
- 6 GB 层需要第 4 步中的离线潜变量和文本缓存。

这些层将 DiT 保留在主机内存中：选择之前请先查看[主机 RAM](configs.md#host-ram)。

4352 个 token 可容纳一对 1 MP 图文：4096 个图像 token，加上最多 256 个 token 的描述文本。更长的描述文本会超出预算，因此基础配置的 `OVERLONG = "drop_tokens"` 会保留该样本图像 token 中的 `4352 - text tokens` 个（按种子抽取，[data.md](../concepts/data.md#resolution-batching-over-long-samples)）。如果要在长描述文本下保留完整图像，请降低面积：在潜变量缓存处理之前，在训练配置中设置 `--set TARGET_AREA=802816`（896^2）或 `--set TARGET_AREA=589824`（768^2）（目标尺寸是潜变量键的一部分）。

<a id="sft-weights"></a>
## SFT 权重

要从 ComfyUI 文件夹在 MOSS-Image-SFT 上训练，请把
`diffusion_models/moss_image_sft_int8_convrot.safetensors` 下载到基础文件旁边，并设置
`--set VARIANT=sft`（DiT 默认值跟随 `VARIANT`；文本编码器和 VAE 共用）。在 HF 布局中：

1. 下载：`hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir
   models/MOSS-Image-SFT`。
2. 设置 SFT 架构预设和权重：`--set VARIANT=sft --set
   WEIGHTS=models/MOSS-Image-SFT`。
3. 如果该文件夹只包含 transformer，请将编码器指向基础模型的下载目录：`--set TEXT_ENCODER=models/MOSS-Image-Base/text_encoder --set
   VAE=models/MOSS-Image-Base/vae`（并在数据配置的 `TEXT_ENCODER_KWARGS` / `IMAGE_ENCODER_KWARGS` 中使用相同路径）。

<a id="fp8-compute"></a>
## fp8 计算

在 sm89+（4090、50xx）上，冻结的基础模型和文本编码器可用 fp8 运行 GEMM：

- `--set BASE_PRECISION=fp8_compute --set FP8_DGRAD=16bit`（Blackwell 上为 `mxfp8`；`FP8_RECIPE` 选择 `fp8_compute` 的缩放粒度）；
- 在 `TEXT_ENCODE` 中设置 `"precision": "fp8_compute"`。

此时基础模型每个权重占 1 字节（`FP8_DGRAD="fp8"`：2 字节，使用 fp8 输入梯度）。精度配置项适用于任何内存配置档（[profiles.md](../concepts/profiles.md#precision-knobs)）；格式及测得速度见 [quant-kernels.md](../internals/quant-kernels.md#fp8-compute)。
