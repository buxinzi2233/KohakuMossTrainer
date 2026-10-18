[英文原文](../../concepts/README.md)

<a id="concepts"></a>
# 概念

KohakuMoss 各个部分的工作原理。先阅读 [model.md](model.md) 和
[seqinfo.md](seqinfo.md)，了解批次布局，再阅读你要配置的部分所对应的页面。

| 页面 | 内容 |
| --- | --- |
| [model.md](model.md) | 此处实现的 MOSS-Image transformer：结构、调用约定、检查点、算子和运行器 |
| [seqinfo.md](seqinfo.md) | 打包批次：样本边界、联合序列顺序、位置、词元丢弃 |
| [numerics.md](numerics.md) | 数据类型、条件计算精度、PyTorch 的默认精度设置、`FP16_FIX` 模式、fp16 修复检查点、残差折叠 |
| [adapters.md](adapters.md) | 适配器配置项、LyCORIS 算法、目标选择预设、倍率、导出、推理加载、通过 KohakuMoss 应用器进行绑定、量化基座 |
| [training.md](training.md) | 构建顺序、内核预热与调优窗口、目标函数、时间步采样器、优化器、优化器更新步骤、精度、流水线布局、EMA、驱动器、检查点、回调、关闭 |
| [profiles.md](profiles.md) | 选择内存配置档、`auto`、编码器放置、`STEP_RELEASE`、内存与精度配置项、配置档目录、预估显存峰值 |
| [data.md](data.md) | 数据集条目、数据源、描述文本、分辨率与批次划分、缓存、目录缓存、HostBatch、加载器、文件格式 |
