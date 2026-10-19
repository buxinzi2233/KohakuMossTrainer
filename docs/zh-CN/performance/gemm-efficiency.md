[English](../../performance/gemm-efficiency.md)

<a id="gemm-efficiency-per-card"></a>
# 每张显卡的 GEMM 效率

一次训练步骤中 GEMM 与 flash-attention 内核达到的张量 TFLOP/s，对照两个参考值：

- 时钟峰值：SM 数 x 每时钟每 SM 的 FLOP x 该内核运行期间实测的 SM 时钟；
- 实测上限：同一张显卡上一次大型 cuBLAS GEMM 的速率。

涉及的显卡为 RTX 5090、RTX 3090 和 RTX 2080 Ti，另加 V100 的上限。

<a id="method"></a>
## 方法

**运行：** fp16 的 LoKr 设置（factor 16，full matrix），第 6 步，工作树 v0.1.0。
内存配置档在 RTX 3090 和 2080 Ti 上为 `auto`，在 RTX 5090 上为
`ckpt-resident`。

**逐内核**（`kohakumoss_bench/measure/tflops.py`）：

- **性能分析。** 对 demangle 后名称匹配 `gemm|flash` 的启动执行 ncu application
  replay，范围限定在该次运行的 `PROFILE_STEPS` 窗口内，并使用
  `--clock-control none --cache-control none`。
- **FLOP。** HMMA warp 指令数（`sm__inst_executed_pipe_tensor_op_hmma.sum`；sm120 上
  为 `..._subpipe_hmma_op_hmma` 名称）x 一条指令的 FLOP 数，后者从内核的 mma shape
  读取：`16816` 为 4096，`1688` 为 2048。其他名称按该架构的 fp16 shape 取值
  （sm75 上为 2048；sm86、sm89 和 sm120 上为 4096）。
- **交叉校验。** fp16 -> fp32 张量算子计数器除以指令 FLOP 数，在三张显卡的每次
  启动上都是 1.00。
- **峰值。** SM 数 x 每时钟每 SM 的 FLOP（见下表）x 该次启动自身的 SM 时钟
  （`sm__cycles_elapsed.avg.per_second`）。
- **聚合。** 按内核名称，TFLOP/s 以 FLOP 加权；时钟和峰值以时间加权。

**上限**（`kohakumoss_bench/measure/ceiling.py`）：

- `torch.mm` 8192^3，fp16 和 bf16，PyTorch 默认的 fp32 累加，分别允许和禁止降精度
  归约。
- 预热 3 s，然后取 20 次计时启动的中位数。
- SM 时钟、功耗和温度来自 nvidia-smi 的 20 ms 采样，覆盖预热循环的最后 1.5 s 以及
  各次计时启动。
- 实际启动的内核由 torch.profiler 记录。

```
python -m kohakumoss_bench.measure.tflops run --config <train config with PROFILE_STEPS> \
    --out <dir> --sm 86 --sms 82 --ncu "sudo -n /usr/local/cuda/bin/ncu"
python -m kohakumoss_bench.measure.tflops report --report <dir>/tf.ncu-rep --sm 86 --sms 82
python -m kohakumoss_bench.measure.ceiling --out ceiling.json
```

`--inst-metric`、`--ops-metric`、`--flop-per-inst` 和 `--flop-per-clk` 用于为表外的
架构设置这些值。输出文件为：

- `tflops.json`：每次启动、逐内核的行、总计，以及交叉校验比值；
- `tflops.md`：表格。

<a id="peak-per-clock"></a>
## 每时钟峰值

稠密 fp16 输入 / fp32 累加的每时钟每 SM FLOP（`FLOP_PER_CLK`）：

| 架构 | 显卡 | FLOP/clk/SM | 它复现的数据表数值 | 实测上限 |
| --- | --- | --- | --- | --- |
| sm70 | Tesla V100 | 1024 | 125 TFLOP/s = 80 个 SM 在 1.53 GHz | 937 (91.5 %) |
| sm75 | RTX 2080 Ti | 512 | 53.8 TFLOP/s = 68 个 SM 在 1.545 GHz | 490 (95.7 %) |
| sm86 | RTX 3090 | 512 | 71 TFLOP/s = 82 个 SM 在 1.695 GHz | 502 (98.0 %) |
| sm89 | RTX 4090 | 512 | 165.2 TFLOP/s = 128 个 SM 在 2.52 GHz | 未测量 |
| sm120 | RTX 5090 | 512 | 209.5 TFLOP/s = 170 个 SM 在 2.407 GHz | 487-489 (95.1-95.5 %) |

GeForce 显卡的 fp32 累加速率是 fp16 累加的一半。数据表数值是该峰值在 boost 时钟下的
取值。负载下时钟与 boost 时钟不同，因此下文每一行都使用该内核运行期间实测的时钟。

<a id="measured-ceilings"></a>
## 实测上限

| 显卡 | dtype | TFLOP/s | SM 时钟 (MHz) | 功耗 (W) / 上限 | FLOP/clk/SM | 内核 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | fp16 | 223.0 | 2685 | 575 / 575 | 489 | `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_nn` |
| RTX 5090 | bf16 | 230.5 | 2782 | 575 / 575 | 487 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_nn` |
| RTX 3090 | fp16 | 71.6 | 1740 | 369 / 370 | 502 | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` |
| RTX 3090 | bf16 | 72.2 | 1755 | 368 / 370 | 501 | `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` |
| RTX 2080 Ti | fp16 | 52.5 | 1575 | 250 / 250 | 490 | `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` |
| RTX 2080 Ti | bf16 | 6.9 | 1500 | 249 / 250 | 68 | `magma_sgemmEx_kernel`（sm75 上没有 bf16 张量路径） |
| Tesla V100 SXM2 16 GB | fp16 | 97.3 | 1297 | 297 / 300 | 937 | `cutlass_70_tensorop_f16_s884gemm_relu_f16_256x128_nn` |
| Tesla V100 SXM2 16 GB | bf16 | 10.6 | 1440 | 298 / 300 | 92 | `magma_sgemmEx_kernel`（sm70 上没有 bf16 张量路径） |

- 每个持续运行的 GEMM 都让显卡处在功耗上限上，而上限决定了时钟。
- 允许或禁止降精度归约，任何一行的变动都不超过 3 %。
- 每个 16 位内核都以 fp32 累加（cuBLAS / CUTLASS 名称中的 `s`）。

<a id="per-kernel"></a>
## 逐内核

列说明：

| 列 | 含义 |
| --- | --- |
| share | 该步骤 GEMM + flash 时间的占比 |
| clock | 该内核按时间加权的 SM 时钟，GHz |
| % peak | TFLOP/s 相对该时钟下的时钟峰值 |
| % datasheet | TFLOP/s 相对数据表数值 |
| % ceiling | TFLOP/s 相对上文的 fp16 上限 |

<a id="rtx-5090-170-sms"></a>
### RTX 5090（170 个 SM）

| 内核 | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn` | 26.5 | 234.6 | 2.865 | 94.1 | 112 | 105 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_tn` | 20.6 | 242.4 | 2.908 | 95.8 | 116 | 109 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_nn` | 18.0 | 239.0 | 2.881 | 95.3 | 114 | 107 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_nn` | 5.2 | 238.9 | 2.911 | 94.3 | 114 | 107 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x128_64x3_nt` | 1.5 | 203.0 | 2.689 | 86.7 | 97 | 91 |
| `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_64x1_tn` | 1.4 | 85.6 | 2.690 | 36.6 | 41 | 38 |
| `flash_fwd_kernel` | 10.4 | 230.6 | 3.112 | 85.1 | 110 | 103 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 12.4 | 238.4 | 3.049 | 89.8 | 114 | 107 |
| 所有 GEMM | 76.6 | 231.0 | 2.868 | 92.5 | 110 | 104 |
| 所有 GEMM + flash | 100 | 230.4 | 2.916 | 90.8 | 110 | 103 |

<a id="rtx-3090-82-sms"></a>
### RTX 3090（82 个 SM）

| 内核 | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn` | 25.1 | 69.1 | 1.751 | 94.0 | 97 | 97 |
| `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn` | 19.4 | 71.2 | 1.751 | 96.9 | 100 | 99 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn` | 14.5 | 71.8 | 1.751 | 97.7 | 101 | 100 |
| `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_nn` | 7.2 | 71.0 | 1.751 | 96.6 | 100 | 99 |
| `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nt` | 5.5 | 26.2 | 1.754 | 35.6 | 37 | 37 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 2.0 | 65.3 | 1.750 | 88.9 | 92 | 91 |
| `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_32x1_tn` | 1.4 | 22.0 | 1.744 | 30.0 | 31 | 31 |
| `flash_fwd_kernel` | 10.1 | 65.4 | 1.751 | 89.0 | 92 | 91 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 12.3 | 67.1 | 1.751 | 91.2 | 95 | 94 |
| 所有 GEMM | 77.3 | 66.0 | 1.751 | 89.7 | 93 | 92 |
| 所有 GEMM + flash | 100 | 65.8 | 1.751 | 89.5 | 93 | 92 |

<a id="rtx-2080-ti-68-sms"></a>
### RTX 2080 Ti（68 个 SM）

sm75 上的 attention 是 TileLang 内核，`gemm|flash` 过滤器不会选中它。

| 内核 | share % | TFLOP/s | clock | % peak | % datasheet | % ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 40.1 | 59.1 | 1.756 | 96.6 | 110 | 113 |
| `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_tn` | 20.8 | 56.8 | 1.741 | 93.7 | 106 | 108 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` | 16.7 | 58.3 | 1.756 | 95.4 | 108 | 111 |
| `turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn` | 13.6 | 58.2 | 1.734 | 96.4 | 108 | 111 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nt` | 3.0 | 43.5 | 1.754 | 71.2 | 81 | 83 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_tn` | 2.4 | 47.1 | 1.755 | 77.1 | 88 | 90 |
| `cutlass_75_tensorop_f16_s1688gemm_relu_f16_64x64_tn` | 0.6 | 20.5 | 1.756 | 33.5 | 38 | 39 |
| 所有 GEMM | 100 | 56.9 | 1.749 | 93.4 | 106 | 108 |

<a id="reading"></a>
## 解读

- **大型 GEMM 已达到时钟峰值。** 在三张显卡上它们都运行在时钟峰值的 93.7-97.7 %，
  flash 内核为 85-91 %。在 RTX 3090 上，四个大型 GEMM（占 GEMM + flash 时间的
  66 %）运行在实测上限的 97-100 %，时钟与上限时的 1.74-1.75 GHz 相同。
- **功耗上限决定了训练时钟。** 持续运行的 GEMM 会让显卡一直处于功耗上限：RTX 5090
  运行在 2.65-2.78 GHz，2080 Ti 为 1.53-1.58 GHz。在 ncu 重放中，每次启动都在空闲
  间隙之间单独运行，时钟更高（5090：2.87-3.11 GHz；2080 Ti：1.75 GHz）。这就是那里
  的 "% ceiling" 和 "% datasheet" 超过 100 的原因。"% peak" 是不依赖时钟的数值。
- **较弱的内核：**
  - RTX 3090 上的 `nt` 256x128 GEMM：35.6 %，占 5.5 % 的时间；
  - 小型 `wmma` 32x32 GEMM：在 RTX 3090 和 5090 上为 30-37 %；
  - 2080 Ti 上的 `nt` / 分阶段 `tn` 变体：71-77 %。
- **如何解读 ncu 的张量流水线数值。** `sm__pipe_tensor_op_hmma_cycles_active` 在 sm86
  和 sm75 上读出的值是 "% peak" 的一半：处于时钟峰值 93.7-97.7 % 的内核读出
  46.8-48.8 %。因此在这些显卡上，一步 fp32 累加 GEMM 的该数值最高约为 50 %。
  在 sm120 上该指标与 "% peak" 读数相同（大型 GEMM 上为 94.8-96.2 %）。
  - [基准测试的 ncu 轮次](bench-matrix.md#ncu)的全步骤张量活跃度（对该步骤的每个内核
    按持续时间加权）在 RTX 3090 上为 39.6 %，在 2080 Ti 上为 36.7 %。这分别是
    fp32 累加 GEMM 所能达到数值的 79 % 和 73 %。
  - 该数值在 RTX 5090 上为 70.0 %。
  - 在 2080 Ti 上，TileLang attention 内核占该步骤内核时间的 32.5 %，张量活跃度为
    30.3 %。
