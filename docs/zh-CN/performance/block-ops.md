[English](../../performance/block-ops.md)

<a id="block-ops-kernels-vs-pytorch-per-gpu"></a>
# 块算子：各 GPU 上内核与 PyTorch 的对比

各 GPU 上 MOSS 块算子（归一化、残差、qk-norm + rope、SiTU、损失、融合线性层）的前向 + 反向设备时间，单位为 ms：将每个内核与 `torch.compile` 及 eager 组合实现进行对比，并列出按形状选择时绑定的候选实现。

| 列 | 含义 |
| --- | --- |
| TileLang, Triton | 后端的内核 |
| compile | 对 eager 参考实现执行动态形状的 `torch.compile`（compile 候选实现） |
| eager | eager 组合实现 |
| bound | 按形状选择时绑定的候选实现；`kernel` 是该架构的首选后端，"a / b" 表示不同运行可能绑定其中任一个 |

- **方法：** 取 20 次中的最佳值，刷新 L2（`scripts/bench/kernel/`、`kohakumoss_kernels/bench/blockops.py`）；采用 MOSS 形状（D = 3072，FFN 2 x 8192，24 个头 x 128）。
- **选择：** 按形状选择机制对每个问题的每个候选实现计时，并绑定最快者，因此算子绝不会运行比它测得的最快实现更慢的候选实现；如果某行的候选实现差距只有几个百分点，不同运行可能绑定其中任一个。
- **精度：** 列出的每个内核在每一行都通过了相对 PyTorch 的误差统计检查（[fp16.md](../internals/fp16.md#eager-reference-and-error-statistics)）。

<a id="sm75-tesla-t10-fp16"></a>
## sm75（Tesla T10），fp16

| 算子 | 用例 | TileLang | Triton | compile | eager | bound |
| --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.512 | 0.425 | 0.553 | 6.73 | kernel / triton |
| rms_norm_mod | N=4400，全部梯度 | 0.680 | 0.452 | 1.319 | 10.18 | triton |
| rms_norm | N=4400 | 0.651 | 0.442 | 1.084 | 5.40 | kernel / triton |
| rms_norm_mod | N=16400 B=4, dx | 1.830 | 1.423 | 2.332 | 24.98 | triton |
| rms_norm_mod | N=16400 B=4，全部梯度 | 1.614 | 1.430 | 5.114 | 43.75 | triton |
| rms_norm | N=16400 | 1.651 | 1.473 | 3.837 | 19.92 | triton |
| gated_residual | N=4400，门控冻结 | 0.524 | 0.404 | 0.451 | 3.31 | triton / compile |
| gated_residual | N=4400，门控参与训练 | 0.680 | 0.482 | 0.561 | 5.18 | triton |
| gated_residual | N=16400 B=4，门控冻结 | 1.950 | 1.425 | 1.669 | 10.65 | triton |
| gated_residual | N=16400 B=4，门控参与训练 | 2.287 | 1.698 | 1.947 | 16.58 | triton |
| qk_norm_rope | N=4396，权重冻结 | 4.110 | 0.765 | 3.692 | 15.40 | triton |
| qk_norm_rope | N=4396，权重参与训练 | 4.157 | 0.858 | 4.942 | 16.61 | triton |
| qk_norm_rope | N=17381 B=4，权重冻结 | 16.02 | 3.228 | 14.42 | 60.44 | triton |
| qk_norm_rope | N=17381 B=4，权重参与训练 | 16.09 | 3.595 | 19.82 | 65.06 | triton |
| situ_glu | N=4400 | 1.591 | - | 2.659 | 25.89 | kernel |
| token_mse | N=4096 | 0.108 | 0.100 | 0.025 | 0.104 | compile |
| token_mse | N=15264 | 0.178 | 0.072 | 0.070 | 0.442 | compile / triton |
| linear_situ | N=17381 | 74.9 | - | 80.3 | 185.6 | kernel |
| linear_gated_residual，FFN 下投影 | N=17381 | 36.0 | - | 36.1 | 46.1 | kernel / compile |
| linear_gated_residual，输出投影 | N=17381 | 14.1 | - | 13.9 | 21.4 | kernel / compile |
| linear_qk_rope | N=17381 | 38.7 | - | 45.0 | 80.8 | kernel |

<a id="sm70-tesla-v100-32g-fp16"></a>
## sm70（Tesla V100 32G），fp16

这里没有 Triton 列：它在 sm70 上的测量运行于一张被其他任务占用的 GPU 上，因此无法与其他列比较，故不列出。该次运行的按形状选择机制将 Triton 纳入候选实现进行了计时，所以 `bound` 列尚待在空闲 V100 上重新测量。

| 算子 | 用例 | TileLang | compile | eager | bound |
| --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.489 | 0.486 | 2.59 | 待重新测量 |
| rms_norm_mod | N=4400，全部梯度 | 1.190 | 0.660 | 3.37 | 待重新测量 |
| rms_norm | N=4400 | 1.977 | 0.578 | 2.13 | 待重新测量 |
| rms_norm_mod | N=16400 B=4, dx | 0.991 | 1.196 | 9.30 | 待重新测量 |
| rms_norm_mod | N=16400 B=4，全部梯度 | 1.253 | 2.243 | 12.09 | 待重新测量 |
| rms_norm | N=16400 | 2.407 | 1.452 | 7.57 | 待重新测量 |
| gated_residual | N=4400，门控冻结 | 0.733 | 0.516 | 1.14 | 待重新测量 |
| gated_residual | N=4400，门控参与训练 | 0.866 | 0.525 | 1.49 | 待重新测量 |
| gated_residual | N=16400 B=4，门控冻结 | 0.843 | 1.060 | 4.07 | 待重新测量 |
| gated_residual | N=16400 B=4，门控参与训练 | 1.064 | 1.087 | 5.36 | 待重新测量 |
| qk_norm_rope | N=4396，权重冻结 | 2.230 | 2.908 | 6.07 | 待重新测量 |
| qk_norm_rope | N=4396，权重参与训练 | 2.279 | 3.344 | 6.69 | 待重新测量 |
| qk_norm_rope | N=17381 B=4，权重冻结 | 6.932 | 10.96 | 22.99 | 待重新测量 |
| qk_norm_rope | N=17381 B=4，权重参与训练 | 7.021 | 12.65 | 24.84 | 待重新测量 |
| situ_glu | N=4400 | 0.696 | 1.912 | 9.50 | 待重新测量 |
| situ_glu | N=16400 | 2.851 | 6.926 | 35.19 | 待重新测量 |
| token_mse | N=4096 | 0.678 | 0.258 | 0.307 | 待重新测量 |
| token_mse | N=15264 | 0.666 | 0.346 | 0.389 | 待重新测量 |

<a id="sm86-rtx-3090-triton-fp16--bf16"></a>
## sm86（RTX 3090），Triton，fp16 / bf16

| 算子 | 用例 | Triton fp16 | compile fp16 | Triton bf16 | compile bf16 | eager fp16 | bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.587 | 0.391 | 0.578 | 0.391 | 2.97 | compile |
| rms_norm_mod | N=4400，全部梯度 | 1.158 | 0.554 | 1.232 | 0.588 | 4.72 | compile |
| rms_norm | N=4400 | 1.159 | 0.499 | 1.198 | 0.499 | 2.39 | compile |
| rms_norm_mod | N=16400 B=4, dx | 0.896 | 1.145 | 1.163 | 1.049 | 10.78 | kernel |
| rms_norm_mod | N=16400 B=4，全部梯度 | 1.276 | 1.664 | 1.299 | 1.646 | 17.03 | kernel |
| rms_norm | N=16400 | 1.012 | 1.276 | 1.184 | 1.278 | 8.63 | kernel |
| gated_residual | N=4400，门控冻结 | 0.662 | 0.291 | 0.172 | 0.198 | 1.26 | kernel / compile |
| gated_residual | N=4400，门控参与训练 | 0.914 | 0.523 | 0.877 | 0.504 | 2.00 | compile |
| gated_residual | N=16400 B=4，门控冻结 | 0.950 | 0.910 | 0.965 | 0.828 | 4.56 | kernel / compile |
| gated_residual | N=16400 B=4，门控参与训练 | 1.025 | 0.836 | 1.019 | 0.979 | 7.50 | kernel / compile |
| qk_norm_rope | N=4396，权重冻结 | 0.865 | 1.921 | 0.882 | 2.005 | 6.80 | kernel |
| qk_norm_rope | N=4396，权重参与训练 | 1.073 | 2.332 | 1.230 | 2.343 | 7.46 | kernel |
| qk_norm_rope | N=17381 B=4，权重冻结 | 1.402 | 7.188 | 1.399 | 7.415 | 26.06 | kernel |
| qk_norm_rope | N=17381 B=4，权重参与训练 | 1.581 | 8.798 | 1.591 | 8.680 | 28.21 | kernel |
| situ_glu | N=4400 | 0.813 | 1.493 | 0.851 | 1.617 | 11.12 | kernel |
| situ_glu | N=16400 | 2.940 | 5.399 | 2.825 | 5.660 | 41.25 | kernel |
| token_mse | N=4096 | 0.540 | 0.327 | 0.541 | 0.323 | 0.44 | compile |
| token_mse | N=15264 | 0.031 | 0.289 | 0.462 | 0.330 | 0.45 | kernel / compile |

<a id="sm120-rtx-5090-triton-fp16--bf16"></a>
## sm120（RTX 5090），Triton，fp16 / bf16

| 算子 | 用例 | Triton fp16 | compile fp16 | Triton bf16 | compile bf16 | bound fp16 |
| --- | --- | --- | --- | --- | --- | --- |
| rms_norm_mod | N=4400, dx | 0.306 | 0.273 | 0.306 | 0.252 | compile |
| rms_norm_mod | N=4400，全部梯度 | 0.404 | 0.472 | 0.460 | 0.473 | compile |
| rms_norm | N=4400 | 0.393 | 0.242 | 0.440 | 0.284 | compile |
| rms_norm_mod | N=16400 B=4, dx | 0.368 | 0.363 | 0.400 | 0.348 | compile |
| rms_norm_mod | N=16400 B=4，全部梯度 | 0.443 | 0.872 | 0.441 | 0.871 | kernel |
| rms_norm | N=16400 | 0.462 | 0.530 | 0.422 | 0.492 | compile |
| gated_residual | N=4400，门控冻结 | 0.305 | 0.250 | 0.325 | 0.235 | compile |
| gated_residual | N=4400，门控参与训练 | 0.440 | 0.311 | 0.436 | 0.306 | compile |
| gated_residual | N=16400 B=4，门控冻结 | 0.343 | 0.524 | 0.341 | 0.371 | kernel |
| gated_residual | N=16400 B=4，门控参与训练 | 0.460 | 0.534 | 0.463 | 0.536 | compile |
| qk_norm_rope | N=4396，权重冻结 | 0.289 | 0.692 | 0.296 | 0.697 | kernel |
| qk_norm_rope | N=4396，权重参与训练 | 0.419 | 0.855 | 0.425 | 0.812 | kernel |
| qk_norm_rope | N=17381 B=4，权重冻结 | 0.662 | 2.520 | 0.663 | 2.593 | kernel |
| qk_norm_rope | N=17381 B=4，权重参与训练 | 0.729 | 3.218 | 0.731 | 3.057 | kernel |
| situ_glu | N=4400 | 0.360 | 0.580 | 0.348 | 0.577 | kernel |
| situ_glu | N=16400 | 1.261 | 1.752 | 1.258 | 1.726 | kernel |
| token_mse | N=4096 | 0.283 | 0.230 | 0.277 | 0.233 | compile |
| token_mse | N=15264 | 0.288 | 0.253 | 0.295 | 0.264 | compile |
| linear_situ | N=17381 | 15.84 | 17.07 | 16.55 | 16.73 | kernel / compile |
| linear_gated_residual，FFN 下投影 | N=17381 | 7.80 | 8.01 | 7.65 | 7.86 | kernel |
| linear_gated_residual，输出投影 | N=17381 | 3.09 | 3.30 | 3.05 | 3.26 | kernel |
| linear_qk_rope | N=17381 | 6.51 | 10.23 | 6.41 | 10.06 | kernel |

<a id="where-torchcompile-wins"></a>
## torch.compile 胜出的情况

- sm86 / sm120 上单样本的归一化、残差和损失（N 为数千 token）：compile 更快，选择机制在这些情况下绑定 compile（sm70：上表中 compile 也胜过 TileLang；其绑定结果尚待重新测量）。
  - 这些行受主机端限制：在 5090 上，Triton `rms_norm_mod` 的前向 + 反向消耗约 430 us 主机时间，而设备时间约为 170 us，compile 的主机时间约为 350 us。
  - Triton 路径跳过自定义算子分派（`_common.eager_function`），绑定其已调优的配置，按 `seg` 张量缓存段边界，并将参数梯度尾处理与段求和放在一次启动中运行。剩下的开销是 autograd 引擎、每次调用的两个 Python `autograd.Function` 节点和两次 Triton 启动，而 Inductor 仅有一张生成的图。
  - 要缩小差距，需要减少每个块的启动次数（融合相邻算子）或捕获整个步骤，而不是更快的内核。
- `token_mse`：在每张显卡上都受启动开销限制；compile 和 Triton 互有胜负。
- TileLang 的逐行内核通过逐元素加载，在 sm70 / sm75 上达到 DRAM 带宽的 25-60 %；前向和仅计算 dx 的反向还各有一个 `vec` 变体（通过分块复制将一行暂存到共享内存中），调优器会将其与这些内核进行计时对比。
