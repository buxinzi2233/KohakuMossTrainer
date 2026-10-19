[English](../../performance/README.md)

<a id="performance"></a>
# 性能

| 页面 | 内容 |
| --- | --- |
| [attention.md](attention.md) | 各 GPU 上变长注意力与 PyTorch SDPA 的对比 |
| [block-ops.md](block-ops.md) | 各 GPU 上归一化 / 残差 / qk-rope / SiTU / 损失 / 融合线性层与 PyTorch eager 及 compile 的对比 |
| [offload-synthetic.md](offload-synthetic.md) | 在 MOSS 规模的合成块上检查卸载选项及硬件极限，涵盖 V100 和 RTX 4090 |
| [gemm-efficiency.md](gemm-efficiency.md) | 一次训练步骤中 GEMM / flash 内核达到的 TFLOP/s，对照时钟峰值与实测 cuBLAS 上限，涵盖 RTX 5090 / 3090 / 2080 Ti 和 V100 |
| [quant-rotate.md](quant-rotate.md) | 在 DiT 形状下，ConvRot 旋转与旋转 int8 linear 对比普通 int8 linear，分离与融合，各 GPU |
| [training-5090.md](training-5090.md) | 一块和四块 RTX 5090 上 LoRA 与全量微调的 tokens/s，5090 配方的设置及其邻近设置 |
| [vae.md](vae.md) | 各 GPU 上按 dtype、compile、分块、批量划分的 VAE 编码 / 解码每张图像耗时（ms）及峰值显存 |
| [bench-matrix.md](bench-matrix.md) | 真实 GPU 基准测试矩阵：每张显卡的速度 / 显存 / 质量 / 推理运行、功能覆盖情况及报告 |
