[English](../../performance/vae.md)

<a id="vae-speed-and-vram-per-dtype"></a>
# VAE：各 dtype 的速度与显存

按 GPU、dtype 和变体列出 FLUX.2 VAE 的编码（像素到潜变量）、解码（潜变量到像素）耗时及峰值内存。选定的默认值与 VAE 精度见 [internals/offload.md#vae](../internals/offload.md#vae)。

表格阅读说明：

| 术语 | 含义 |
| --- | --- |
| ms | 每张图像：对 4 张图像进行 3 次计时重复，取中位数 |
| GiB（峰值已分配量） | 一次重复期间的峰值已分配内存 |
| `eager` | 默认布局 |
| `cl_compile` | channels_last 加 `torch.compile(dynamic=True)`（`FluxVae.accelerate`） |
| `tiled` | diffusers 的分块 / 切片编码与解码 |
| `cl` | 仅 channels_last；它在所有情况下都落后，仅存在于原始 JSON 中 |
| 粗体 | 某 GPU 和图像面积下最快的行（批量 1） |

输入为真实数据集图像，并使用真实权重，中心裁剪至 1 MP（1024²）或 2 MP（1440²）。基准测试：`scripts/bench/vae/vae.py`，写入 `runs/bench/vae/<gpu>.json`；内存不足的配置会记录为 `oom` 行，基准测试随后继续。

<a id="batch-1"></a>
## 批量 1

默认设置（`VAE_ENCODE` 批量 1，`compile='auto'`）：

| GPU | dtype | 变体 | 面积 | 编码 ms | 编码 GiB（峰值已分配量） | 解码 ms | 解码 GiB（峰值已分配量） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | fp32 | eager | 1MP | 102.0 | 2.80 | 131.5 | 3.10 |
| RTX 5090 | fp32 | cl_compile | 1MP | 87.1 | 2.82 | 117.8 | 2.91 |
| RTX 5090 | bf16 | eager | 1MP | 59.3 | 1.67 | 78.7 | 1.94 |
| RTX 5090 | bf16 | tiled | 1MP | 59.5 | 1.67 | 79.0 | 1.94 |
| RTX 5090 | bf16 | cl_compile | 1MP | **39.3** | 1.18 | **53.4** | 1.28 |
| RTX 5090 | fp16 | cl_compile | 1MP | 39.7 | 1.18 | 53.8 | 1.28 |
| RTX 5090 | fp32 | eager | 2MP | 218.7 | 5.26 | 275.0 | 5.86 |
| RTX 5090 | bf16 | eager | 2MP | 126.9 | 3.15 | 166.3 | 3.69 |
| RTX 5090 | bf16 | tiled | 2MP | 151.6 | 1.69 | 241.8 | 1.95 |
| RTX 5090 | bf16 | cl_compile | 2MP | **84.9** | 2.18 | **115.1** | 2.39 |
| RTX 3090 | fp32 | eager | 1MP | 258.4 | 2.77 | 328.5 | 3.07 |
| RTX 3090 | fp32 | cl_compile | 1MP | 243.7 | 2.80 | 330.1 | 2.88 |
| RTX 3090 | bf16 | eager | 1MP | 133.7 | 1.65 | 176.8 | 1.92 |
| RTX 3090 | bf16 | tiled | 1MP | 134.7 | 1.65 | 178.1 | 1.92 |
| RTX 3090 | bf16 | cl_compile | 1MP | **109.0** | 1.16 | **153.4** | 1.26 |
| RTX 3090 | fp16 | cl_compile | 1MP | 108.8 | 1.16 | 154.3 | 1.26 |
| RTX 3090 | fp32 | eager | 2MP | 626.9 | 5.24 | 742.5 | 5.83 |
| RTX 3090 | bf16 | eager | 2MP | 296.6 | 3.13 | 372.6 | 3.67 |
| RTX 3090 | bf16 | tiled | 2MP | 365.1 | 1.66 | 527.3 | 1.93 |
| RTX 3090 | bf16 | cl_compile | 2MP | **243.7** | 2.15 | **340.3** | 2.37 |
| V100 32G | fp32 | eager | 1MP | 327.4 | 2.77 | 387.1 | 2.51 |
| V100 32G | fp32 | cl_compile | 1MP | 354.8 | 3.30 | 424.1 | 3.26 |
| V100 32G | fp16 | eager | 1MP | 143.3 | 1.40 | 192.8 | 1.55 |
| V100 32G | fp16 | tiled | 1MP | 146.6 | 1.40 | 200.1 | 1.55 |
| V100 32G | fp16 | cl_compile | 1MP | **136.9** | 1.16 | **172.2** | 1.26 |
| V100 32G | fp32 | eager | 2MP | 824.8 | 5.24 | 923.8 | 4.72 |
| V100 32G | fp16 | eager | 2MP | 334.3 | 2.65 | 435.5 | 2.94 |
| V100 32G | fp16 | tiled | 2MP | 408.2 | 1.42 | 581.8 | 1.56 |
| V100 32G | fp16 | cl_compile | 2MP | **314.1** | 2.15 | **377.6** | 2.37 |

<a id="batch-4"></a>
## 批量 4

每次调用处理 4 张同尺寸图像；耗时为每张图像的 ms：

| GPU | dtype | 变体 | 面积 | 编码 ms | 编码 GiB（峰值已分配量） | 解码 ms | 解码 GiB（峰值已分配量） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | bf16 | eager | 1MP | 46.4 | 6.18 | 76.6 | 6.17 |
| RTX 5090 | bf16 | cl_compile | 1MP | 43.4 | 4.21 | 51.9 | 4.66 |
| RTX 5090 | bf16 | eager | 2MP | 110.8 | 6.13 | 164.1 | 12.05 |
| RTX 5090 | bf16 | cl_compile | 2MP | 98.8 | 4.18 | 114.4 | 9.07 |
| RTX 3090 | bf16 | eager | 1MP | 118.2 | 6.16 | 232.4 | 6.14 |
| RTX 3090 | bf16 | cl_compile | 1MP | 128.2 | 4.18 | 162.7 | 4.64 |
| RTX 3090 | bf16 | eager | 2MP | 278.2 | 6.10 | 495.2 | 12.02 |
| RTX 3090 | bf16 | cl_compile | 2MP | 287.0 | 4.15 | 357.5 | 9.05 |
| RTX 3090 | fp32 | eager | 2MP | 601.3 | 10.20 | OOM | - |
| V100 32G | fp16 | eager | 1MP | 127.4 | 5.16 | 283.4 | 5.40 |
| V100 32G | fp16 | cl_compile | 1MP | 161.6 | 4.18 | 182.4 | 4.64 |
| V100 32G | fp16 | eager | 2MP | 335.2 | 5.13 | 662.2 | 10.55 |
| V100 32G | fp16 | cl_compile | 2MP | 366.0 | 4.15 | 433.7 | 9.05 |

RTX 3090 在 2 MP、批量 4 下执行 fp32 eager 解码时耗尽了其 24 GB 显存。3090 的 fp32 tiled 行缺失：该次运行在第一次内存不足时停止了。
