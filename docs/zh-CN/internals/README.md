[English](../../internals/README.md)

<a id="internals"></a>
# 内部实现

各部分内部如何工作：面向贡献者，以及想了解某个配置项在设备上有何作用的用户。请先从[概念](../concepts/README.md)和[指南](../guides/README.md)开始；再到这里了解设计及其成本。

| 领域 | 页面 | 内容 |
| --- | --- | --- |
| 内核 | [kernel-dev.md](kernel-dev.md) | 编写和注册内核后端 |
| | [tuning.md](tuning.md) | 规划器、计时候选短名单、调优表、预热、调优窗口 |
| | [fp16.md](fp16.md) | 块算子的精度规则、误差统计、按问题选择内核或编译路径 |
| | [quant-kernels.md](quant-kernels.md) | int8 / fp8 仅权重量化的冻结基座 GEMM；fp8 / MXFP8 计算（格式、配方、内核、选择） |
| 内存 | [offload.md](offload.md) | 激活卸载、权重流式传输、梯度卸载、CPU 优化器、文本编码器放置 |
| | [sampling-memory.md](sampling-memory.md) | 一次 generate 调用在各阶段（文本编码、采样、解码）在设备上保留的内容，以及使采样显存占用不高于训练的放置方案 |
| | [checkpoints.md](checkpoints.md) | 流式检查点写入、内存和磁盘成本 |
| 文本编码器 | [text-encoder.md](text-encoder.md) | 文本编码器算子和内核、批量 / 打包编码、TextEncoderPlan、提前编码、量化 |
| 适配器 | [adapter-kinds.md](adapter-kinds.md) | 各算法的适配器记录 |
| | [adapter-binding.md](adapter-binding.md) | 绑定的适配器前向函数、内核算子、自动调优算子、前向包装器 |
| 多 GPU | [distributed-launch.md](distributed-launch.md) | 进程启动、存储与后端（多 GPU 训练仅支持 Linux；CPU gloo 自检也可在 Windows 上运行） |
