[English](../../guides/interfaces.md) | [文档索引](../README.md)

<a id="external-interface-reference"></a>
# 外部接口参考

本页按当前源码核对（2026-10-23），面向 UI、启动器和 Python 调用方；不是 HTTP/RPC API，也不是版本化协议承诺。配置本身是可执行 Python，只加载可信文件。

已有的 [配置](configs.md)、[脚本目录](scripts.md)、[推理](inference.md)、[日志](logging.md)、[检查点](../internals/checkpoints.md) 保留详细说明；这里补齐跨进程契约，不复制全部训练参数。`plans/ui/interface.md` 是早期调研快照，其中的数量、性能实测、`board` 后端与部分默认值不能作为当前接口规范。

<a id="1-command-entry-points"></a>
## 1. 命令入口

[pyproject.toml](../../../pyproject.toml) 注册 **3 个**本项目命令；`kogine` 由依赖 `kohaku-engine` 提供，不是本项目注册的命令。脚本的标准调用形式：

```bash
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set LR=2e-4
```

从仓库根目录运行；配置路径和普通相对路径参数按进程工作目录解释，`use_config` 的相对引用按配置文件目录解释（见[配置](configs.md)）。安装包中脚本位置由 `cli.config.validate.repo_root()` 寻找 `kohakumoss/bundle/`，`script_ref()` 返回 bundle 下的绝对脚本路径，不要假设安装后仍存在仓库相对脚本。依赖自身额外的 `kogine` 选项以已安装版本为准。

<a id="kohakumoss-config"></a>
### kohakumoss-config

源码：[cli/config/main.py](../../../src/kohakumoss/cli/config/main.py) 的 `build_parser`、`build_from_parser`、`main`。

| 子命令 | 参数与默认值 | 输出 / 行为 |
| --- | --- | --- |
| `new train` | 必填 `--weights`、`--data`；`--source=kohya`、`--algo=lora`、`--gpus=1`、`--name=my-lora`、`--steps=0`（沿用 base） | 写 `<name>.py` 和 `<name>_data.py`；不是一个原子文件事务 |
| `new sample` | 必填 `--weights`；可重复 `--adapter path[:mult]`，默认空；`--name=sample`、`--samples=runs/samples` | 写采样配置；无 prompt 时写入占位提示词并警告 |
| 两类 `new` 共用 | 可重复 `--prompt`，默认空；`--vram`、`--precision` 默认检测硬件，后者仅 `bf16/fp16`；`--vram-tolerance=0.05`；`--out=""`；`--force` 默认关闭 | `--vram` 为 GiB 或 `auto`，sample 不接受 `auto`；无 GPU 且未同时指定 vram/precision 则失败 |
| `new --from BASE OUT` | BASE 是 recipe 名或训练配置路径；`--force` 默认关闭 | 生成继承 base 的完整可编辑训练配置，不是原样复制 |
| `wizard` | 交互问答，无批处理参数 | 生成训练和数据配置；覆盖需确认 |
| `list [--all]` | `--all` 默认关闭 | 文本预设/显存表，无 JSON 输出协议 |
| `check CONFIG...` | 至少一个配置路径 | 每个打印 `ok` 或 `invalid` 与问题列表；全部无问题 exit 0，否则 exit 1 |

`--out` 未设：checkout 写入仓库 `configs/user/`，bundle 安装写入当前目录。已存在文件默认拒绝；`--force` 才覆盖。创建配置时 `blocking()` 仅按问题文本是否包含 `does not exist` 过滤缺路径问题并警告；`check` 和实际训练不作这个豁免。配置解析/I/O 异常直接传播，不保证总能返回问题列表。参数语法错误由 argparse exit 2；正常返回为 0，字符串 `SystemExit` 为失败（常规 Python CLI exit 1）。

<a id="doctor--tune"></a>
### doctor / tune

| 命令 | 参数与默认值 | 结果与副作用 |
| --- | --- | --- |
| `kohakumoss-doctor` | `--json=""`、`--memory=False`、`--clear-failures=False` | 打印环境报告；指定 JSON 路径则覆盖写入，不创建父目录；clear-failures 会清除记录的后端失败，不是只读 |
| `kohakumoss-tune` | `--device=cuda`、`--dtype=bf16`（或 fp16）、`--sizes 1024x1024`（可多个 HxW）、`--variant=base`、`--text-tokens=512`、`--batch=4`、`--backward=False`、`--trainable=adapter`（或 full） | 实际执行设备内核计时，写每卡 tuning 表；不是环境检查，也不只是解析配置 |

源码：[doctor.main/report](../../../src/kohakumoss/cli/doctor.py)、[tune.main/tune](../../../src/kohakumoss/cli/tune.py)。doctor 报告包含 `os/os_release/python/torch_cuda/packages/devices/missing/msvc/compilers/kernels/backend_status/backend_failures`，`--memory` 增加 `memory_plans`。`missing` 非空或报告含失败并不会自动令命令非零退出；调用方须检查字段。不要把报告中部分路径脱敏理解为所有数据都经过保密审查。tune 的 Python 返回值是本次运行的问题数量，CLI 不将这个数量作为退出码；异常仍传播。

<a id="kogine-scripts-and-key-defaults"></a>
### kogine 脚本与主要默认值

参数为模块顶层大写 knob，用 `--set KEY=VALUE` 重复覆盖。配置、覆盖值需要按脚本 schema 转换；不能把字符串当作已验证的布尔/数值。完整脚本集合见[脚本目录](scripts.md)。

| 入口 | 集成时主要参数（脚本默认，不是 recipe 覆盖值） |
| --- | --- |
| [train/moss.py](../../../scripts/train/moss.py) `main` | `WEIGHTS=""`、`DATASETS=[]` 需配置；`ALGO="lora"`、`LR=1e-4`、`STEPS=1000`、`PRECISION="bf16"`、`MEMORY_PROFILE="balanced"`、`DRIVER="native"`、`PARALLEL="none"`、`DEVICES=(0,)` |
| [sample/sample.py](../../../scripts/sample/sample.py) `main/load_items` | `WEIGHTS=""`、`PROMPT=""`、`PROMPTS=[]`、`PROMPTS_FILE=""`、`OUT=""`；`DEVICE="cuda"`、`STEPS=50`、`CFG=3.0`、`SHIFT=5.0`、`BATCH=4`、`SIZES=[(1024,1024)]`、`SEEDS=[]`、`SEED_BASE=0`、`TUNE="off"` |
| [export/export.py](../../../scripts/export/export.py) `main` | `WEIGHTS/TRANSFORMER/ADAPTER/OUT=""`、`VARIANT="base"`、`FORMAT="comfy"`、`STYLE="lora_unet"`、`SAVE_DTYPE="bf16"`、`MULTIPLIER=1.0`、`FUSED_QKV="auto"`、`DEVICE="cpu"`；格式详见[导出](export-to-comfy.md) |

采样按 `PROMPTS`、非空 `PROMPT`、文件内容顺序组装。文件为文本（每个非空行一条）或 JSON 列表（字符串或 `{prompt, negative, size: [H,W], seed}` 对象）。缺省 size 用 `SIZES`，长度不足重复最后一项；缺省 seed 用 `SEEDS[i]`，不足时用 `SEED_BASE+i`；无任何提示词会 `SystemExit`。结果写 `<index>_<seed>.png` 与 `metadata-<first index>.json`，metadata 顶层为 `knobs/settings/run/images`。编号从现有输出最大编号后继续；没有多进程编号锁，不要让多个采样进程共用 OUT。

数据准备主要入口为 `scripts/data/inspect_data.py`、`cache_text.py`、`cache_latents.py`；另有 catalog 构建、cache 合并/清理命令，见[数据指南](datasets.md)与脚本目录。这些入口可能扫描、编码或写缓存，不是纯 schema 查询。

<a id="2-run-and-file-contract"></a>
## 2. run 与文件契约

来源：[training.entry.run/_start/worker](../../../src/kohakumoss/training/entry.py)、[training.knobs.run_name/check](../../../src/kohakumoss/training/knobs.py)、[logging.sinks](../../../src/kohakumoss/logging/sinks.py)。

| knob | 默认值 | 语义 |
| --- | --- | --- |
| `OUTPUT_DIR` / `RUN_NAME` | `"runs/moss"` / `""` | `Path(OUTPUT_DIR) / (RUN_NAME or "run")`；不是自动生成唯一 run ID |
| `LOGGER` / `PROJECT` | `"console"` / `"kohakumoss"` | 内置仅 console、wandb；两者均写 metrics.jsonl；wandb 需依赖且可能联网 |
| `LOG_EVERY` / `METRICS_EVERY` | `10` / `None` | 控制台 / 指标间隔；None 沿用 LOG_EVERY；首个实际执行步（续训点+1）和最后步也触发指标 |
| `RESUME` | `""` | 不恢复；`"latest"` 取当前 run 最新完整 checkpoint；其他字符串作为路径 |
| `SAVE_EVERY` / `SAVE_MINUTES` | `500` / `0` | 每 N 步 / 每 N 分钟保存；0 关闭；分钟条件按 LOG_EVERY 检查 |
| `KEEP_LAST` / `FINAL_CHECKPOINT` | `3` / `"auto"` | 保留最近完整 checkpoint，KEEP_LAST<=0 全留；auto 在周期保存开启时保存正常结束的最终 checkpoint |
| `SAVE_FORMAT` / `SAVE_MERGED` | `"comfy"` / `False` | 正常结束导出；支持 comfy/hf/both/none；不等同于可续训 checkpoint |
| `CHECKPOINT_EXPORT` / `SAVE_FUSED_QKV` | `True` / `"auto"` | checkpoint 也导出；auto 跟随 DiT 权重的 QKV 存法 |

`RUN_NAME` 直接参与 Path 拼接，没有目录沙箱保障；UI 应自行限制路径片段，并固定 cwd 或使用绝对 OUTPUT_DIR。两个独立 run 不可并发写同一目录：没有 run 所有权锁。rank 0 写配置和全局日志，其他 rank 的 logger 是 NullLogger；诊断/每 rank 文件另有自己的写入方式。

| run 内路径 | 内容与更新方式 |
| --- | --- |
| `config.json` | rank 0 每次启动直接覆盖的有效 knob 快照；UTF-8 JSON，indent=2，不能 JSON 化的值转字符串；不是热更新输入 |
| `log.txt` | 日志文本；消费者不要依赖固定消息文本作状态协议 |
| `metrics.jsonl` | UTF-8 逐行追加，每行 flush，详见下一节 |
| `checkpoints/step_XXXXXXXX/` | 可续训状态；先写同名 `.partial/`，完成后 rename |
| `checkpoints/step_XXXXXXXX/state.json` | checkpoint 状态而非实时状态：`step/loader/tokens/samples/joint_sq/text_tokens` 及布局附加字段 |
| `instability/step_XXXXXXXX.json` | non-finite / loss spike 诊断：`step/reason/loss/loss_avg/grad_norm/loss_scale/lr/batch_ids/metrics/nonfinite_grads`，直接写入 |
| `samples/`、`export/`、`ranks/` | 按启用的采样、导出、每 rank 诊断功能出现，不能当作成功标志 |

未续训时 `fresh_run()` 仅把非空 `metrics.jsonl`、`log.txt` 改名为首个可用的 `metrics.N.jsonl`、`log.N.txt`；不会清空旧 checkpoint、样本和导出。`RESUME="latest"` 找不到完整 checkpoint 时返回 None，实际按新 run 处理，**不是错误**。需要强制恢复的启动器应先确认目标存在。

[CheckpointManager](../../../src/kohakumoss/training/checkpoint/manager.py) 只列出名称匹配 `step_\d{8}` 且 `state.json` 与 `trainable.safetensors` 均存在的目录；这只是存在性检查，不是校验和检查。rank 各写 `rng.<rank>.pt`，可选 `optim.<rank>.pt`；rank 0 写 `trainable.safetensors`、可选 `ema.safetensors`、`optim.pt`，最后写 `state.json`，rank 同步后完成目录交接。同 step 已有完整目录会被替换，旧目录按 KEEP_LAST 删除；checkpoint 的 `export/` 在目录提交之后才写，不能仅凭目录出现就认为导出已完成。格式见 [checkpoint.io](../../../src/kohakumoss/training/checkpoint/io.py)、[Parts](../../../src/kohakumoss/training/drivers/common.py)。

显式恢复路径不经 latest 的完整性筛选；文件损坏/缺失可能抛出 I/O、JSON、tensor 加载异常，缺少当前 rank 的 RNG 文件抛 `FileNotFoundError`。不要由 UI 反序列化 `.pt` 来展示状态，也不要修改 checkpoint 内容。

<a id="3-reading-metricsjsonl"></a>
## 3. metrics.jsonl 读取协议

[ConsoleLogger.log/resume_at](../../../src/kohakumoss/logging/sinks.py)、[LogMetrics/Throughput](../../../src/kohakumoss/training/callbacks/monitor.py) 是实际写入方。

| 字段 | 类型 / 单位 / 出现条件 |
| --- | --- |
| `step`, `time` | int 步号；float Unix 秒；正常内置训练行带有这两个字段 |
| `loss`, `grad_norm`, `lr`, `loss_scale` | 训练标量，可能含非有限值 |
| `skipped_steps`, `nonfinite_steps`, `nonfinite_params` | JSON number（实现转 float）；前两项累计，最后一项是当前步参数数量，不是累计 |
| `tokens_per_s`, `samples_per_s`, `step_time`, `tflops` | token/s、sample/s、秒/步、TFLOP/s；`mfu` 仅在已知设备峰值时出现，比例不是百分数 |
| `vram_*`, `te_*`, `vae_*`, `xfer_*` 等 | 可选监测项，取决于硬件和 callback；VRAM 为 GiB，详见[日志](logging.md) |

示意记录（不是本轮实测结果）：

```json
{"step":1,"time":1700000000.0,"loss":0.25,"grad_norm":0.7,"lr":0.0001,"loss_scale":1.0,"skipped_steps":0.0,"nonfinite_steps":0.0,"nonfinite_params":0.0}
```

没有 schema_version、心跳、结束事件或每步必有记录的保证。`json.dumps` 默认允许 `NaN/Infinity/-Infinity`，严格 JSON 消费者需明确处理这些值并显示为缺失/非有限，而不是对整文件执行字符串替换。监测扩展字段应保留/忽略，不能要求每行具有相同键集合。

**读取建议（消费者责任，不是训练器额外保证）：** 保存字节偏移和未完成尾行，只解析完整换行记录；识别文件被替换、缩短、暂时不存在后重开。flush 不是 fsync，也不是多进程事务。启动时 config.json 和诊断 JSON 直接写入，读到未完成 JSON 应重试。

恢复到 step S 时，`resume_at(S)` 关闭流，将 `step>S` 行写到新的 `metrics.N.jsonl`，原文件直接重写为保留行再追加；无后续行则无需重写。其 step 解析失败按 -1 保留，故残缺/坏行不保证被清理。恢复和轮转可能改变文件内容，不能只依赖旧句柄或假设 offset 永远递增。

<a id="4-lifecycle-and-failures"></a>
## 4. 生命周期与错误

- 当前训练接口是启动进程、读取产物和进程退出；没有 HTTP 服务、暂停/继续 RPC、`status.json` 心跳或 knob 文件热加载。`config.json` 和 checkpoint `state.json` 都不是控制通道。
- `training.entry.run(g)` 先收集/归一化并校验完整 knob 集，错误汇总为 `SystemExit("invalid configuration:...")`；不会为任意稀疏 dict 自动补齐所有脚本默认值。随后构建数据/模型并训练。正常进程退出为 0；未捕获异常失败，外层 kogine 的异常封装以安装版本为准。
- [shutdown.install](../../../src/kohakumoss/training/shutdown.py) 仅在主线程安装一次；收到 Python 可处理的 SIGTERM 时抛 `SystemExit(128+signum)`，通常为 143。退出处理清理 multiprocessing 子进程和 forkserver。Windows 强制 TerminateProcess、SIGKILL 等不保证运行 Python 清理。
- SIGTERM 是异常式停止，不是“保存后停止”：默认 [NativeDriver.run](../../../src/kohakumoss/training/drivers/native.py) 的最终 checkpoint 在正常循环之后；[worker](../../../src/kohakumoss/training/entry.py) 的导出也在正常返回路径。已有 finally 尝试资源清理，但不保证中断当前步后生成新的 checkpoint；恢复应选最后完整 checkpoint。
- 多卡/分布式入口可能再派生 rank 进程，`PARALLEL` 的支持边界见[多 GPU](multi-gpu.md)。启动器应管理整个作业而非只轮询一个 PID；日志没有新行并不代表死亡（可能仍在构建、缓存、调优）。以进程实际状态/退出码判断生命周期，以文件判断可恢复产物。

<a id="5-reusable-python-entries"></a>
## 5. 可复用的 Python 入口

这些是当前已有入口，不表示所有内部类都承诺跨版本兼容；UI 优先使用子进程边界，尤其不要在 GUI 主线程导入训练模块、探测设备或执行配置。

| 模块 / 符号 | 输入 → 返回与边界 |
| --- | --- |
| [cli.config.validate](../../../src/kohakumoss/cli/config/validate.py) `config_knobs(config: Path, root=None)` | 返回 `(script: str, knobs: dict)`；docstring 中第一个 `scripts/...py` 决定目标；layer 叠加 base，缺目标或 sweep/generator 会 ValueError |
| 同模块 `compose(root, script, layers, own)` / `config_problems(config, root=None)` | 默认 → 顺序 layers → own；返回 dict / `list[str]`。对非 train/sample/data 脚本 `problems()` 返回空，**不能当作任意脚本校验器**；文件读取/执行异常仍可能传播 |
| [common.script_schema](../../../src/kohakumoss_common/script_schema.py) `static_schema(path: str)` | 不导入目标脚本，返回 `(ScriptSchema, opaque_names)`，支持语句外返回 None；不能静态确定的值在 opaque 集合，不能伪造默认值。自身仍依赖 kohakuengine |
| [training.entry](../../../src/kohakumoss/training/entry.py) `run(g: dict)` | 完整脚本 globals → none/threaded 返回 [TrainState](../../../src/kohakumoss/training/state.py)，进程组启动返回 None；会操作信号/全局调优状态/设备/文件，不是可并发无副作用函数 |
| [kohakumoss_data](../../../src/kohakumoss_data/__init__.py) `DataConfig.from_knobs(k)` / `data_problems(cfg)` / `build_loader(cfg, rank=0, world=1, transform=None, writer=True)` | 分别返回 DataConfig / 问题字符串列表 / Loader。from_knobs 选择数据字段并解析 cache_dir，不自动调用 data_problems；build_loader 可扫描和写 catalog，每 run 只安排一个 writer，其他 rank 设 False |
| [common.registry](../../../src/kohakumoss_common/registry.py) `Registry(family)` / `build(choice, registry, key="name", **defaults)` | registry 名/导入路径/dict/class/buildable/现成对象 → 构建结果；None → None，dict 自带参数优先于 defaults；未知简单名字 KeyError，导入/构造异常传播；仅用于可信扩展 |

`DataConfig` 的 `CACHE_DIR=""` 在 cache 启用时取第一个含 path 的数据集根下 `cache/`；文件/通配符路径取其父目录。两个 tensor cache 均关闭时可保持空；显式 cache_dir 原样保留。完整字段见 [data.config](../../../src/kohakumoss_data/config.py) 和[数据模型](../concepts/data.md)，不要假设 dataclass 默认值全都等于训练脚本默认值。

<a id="mosspipeline"></a>
### MossPipeline

公开导入 `from kohakumoss.inference import MossPipeline, GenerateResult`，实现见 [inference.pipeline](../../../src/kohakumoss/inference/pipeline.py)，完整用法见[推理](inference.md)。

- `MossPipeline.from_pretrained(path, ...) -> MossPipeline`：path 是本地 ComfyUI models 根、pipeline 目录或含 base/sft 的权重根；主要默认值 `variant="base"`、`dtype=torch.bfloat16`、`device="cuda"`、`adapters=()`、`adapter_mode="merge"`、`fp16_fix="none"`、`tune="off"`、`text_encoder=True`、`vae=True`。`transformer`、`text_encoder_file`、`vae_file` 可分别指定组件；规划参数见原有指南。会加载模型并改变进程级调优状态，不保证线程安全。
- `generate(prompts=None, negative_prompts=None, sizes=(1024,1024), steps=50, cfg=3.0, shift=5.0, skip_layers=None, seeds=None, batch=4, cfg_mode="packed", output="pil", prompt_embeds=None, negative_embeds=None, step_callback=None, progress=True)`：尺寸为 **(H,W)**；每个 prompt 一张，默认 seeds 为 `range(n)`，skip_layers 默认取架构预设。提供非空 prompts 或 embeddings，并保证逐项尺寸/种子/负面条件一致；Python 方法不是 CLI knob 校验器，加载/形状/索引错误直接传播，无统一错误对象。
- 返回 `GenerateResult(images, latents, settings)`；`output="pil"` 给 PIL，`"tensor"` 给 `[1,3,H,W]` 的 [-1,1] 张量，`"latent"` 不解码并将 latent 放入 images；未知 output 会 KeyError。`settings` 包含实际调用参数、模型来源与运行统计；方法不自动写图片文件。
- `from_components(backbone, text_encoder=None, vae=None, device=None, memory_plan=None, weight_stream=None)` 共享已有训练组件，不重复加载，也不接管 trainer 的 TUNE。`encode(prompts)` 返回 embedding 列表；`decode(latents)` 返回像素 tensor 列表。

最小配置读取示例（只对可信配置；示例未在本轮执行）：

```python
from pathlib import Path
from kohakumoss.cli.config.validate import config_knobs, problems

script, knobs = config_knobs(Path("configs/user/my-lora.py"))
errors = problems(script, knobs)
if errors:
    raise ValueError("\n".join(errors))
print(script, knobs["OUTPUT_DIR"], knobs["RUN_NAME"])
```

最小采样调用（需已安装依赖、可用设备和本地权重，不会自动准备权重）：

```python
from kohakumoss.inference import MossPipeline

pipe = MossPipeline.from_pretrained("ComfyUI/models", tune="off")
result = pipe.generate(["a lighthouse at dusk"], seeds=[0], progress=False)
result.images[0].save("sample.png")
```

训练示例使用已生成、校验并完成所需缓存的配置；STEPS 是总目标步数，不是续训额外步数：

```bash
kohakumoss-config check configs/user/my-lora.py
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set RUN_NAME=my-lora --set STEPS=1000
# 确认有完整 checkpoint 后恢复同一 run
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set RUN_NAME=my-lora --set RESUME=latest --set STEPS=2000
```

本页验证范围：静态对照命令解析器、脚本默认值、文件写入/恢复路径、公开符号和本地链接；未运行训练、推理、GPU 探测/调优或配置执行。示例参数经过源码核对，不代表实机成功记录。
