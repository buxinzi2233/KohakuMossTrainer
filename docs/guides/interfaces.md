[简体中文](../zh-CN/guides/interfaces.md) | [Docs index](../README.md)

# External interface reference

Checked against the current source on 2026-10-23. This is the integration surface for UIs, launchers and Python callers, not an HTTP/RPC API or a versioned protocol guarantee. Config files execute Python: load trusted files only.

The existing [configs](configs.md), [script catalogue](scripts.md), [inference](inference.md), [logging](logging.md) and [checkpoint](../internals/checkpoints.md) guides remain the detailed references. This page adds cross-process contracts rather than duplicating every training knob. `plans/ui/interface.md` is an earlier research snapshot: its counts, timing measurements, `board` sink and some defaults are not the current specification.

## 1. Command entry points

[pyproject.toml](../../pyproject.toml) registers **three** project commands. `kogine` belongs to the `kohaku-engine` dependency, not this project's console scripts. The standard script invocation is:

```bash
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set LR=2e-4
```

Run from the checkout root. Config arguments and ordinary relative path values are relative to the process cwd; relative `use_config` references are relative to the config file ([configs](configs.md)). In an installed wheel, `cli.config.validate.repo_root()` locates `kohakumoss/bundle/`, and `script_ref()` returns an absolute script path there. Do not assume checkout-relative scripts exist in a wheel install. Additional `kogine` switches depend on the installed engine version.

### kohakumoss-config

Source: [cli/config/main.py](../../src/kohakumoss/cli/config/main.py), `build_parser`, `build_from_parser`, `main`.

| Subcommand | Arguments and defaults | Output / behavior |
| --- | --- | --- |
| `new train` | Required `--weights`, `--data`; `--source=kohya`, `--algo=lora`, `--gpus=1`, `--name=my-lora`, `--steps=0` (keep base) | Writes `<name>.py` and `<name>_data.py`; not an atomic two-file transaction |
| `new sample` | Required `--weights`; repeated `--adapter path[:mult]`, empty by default; `--name=sample`, `--samples=runs/samples` | Writes a sample config; warns and uses a placeholder prompt if none is supplied |
| Both `new` kinds | Repeated `--prompt`, empty by default; hardware-detected `--vram` / `--precision` (bf16 or fp16); `--vram-tolerance=0.05`; `--out=""`; `--force` off | VRAM is GiB or `auto` (sample rejects auto); no GPU without both explicit VRAM and precision is an error |
| `new --from BASE OUT` | Recipe name or train config path; `--force` off | Writes a full editable train config extending the base, not a byte-for-byte copy |
| `wizard` | Interactive questions | Writes train/data configs; prompts before overwriting |
| `list [--all]` | `--all` off | Text preset/memory tables, no JSON output protocol |
| `check CONFIG...` | One or more paths | Prints `ok` or `invalid` and the problems for each; exit 0 if all pass, otherwise 1 |

Without `--out`, a checkout writes under its `configs/user/`; a bundle install uses cwd. Existing files are refused unless `--force` is set. During creation, `blocking()` treats problems containing the literal text `does not exist` as warnings; `check` and actual training do not grant that exemption. Parsing/I/O exceptions propagate and need not become a problem list. Argparse syntax errors exit 2, normal return exits 0, and string `SystemExit` is failure (normally Python CLI exit 1).

### doctor / tune

| Command | Arguments and defaults | Result and side effects |
| --- | --- | --- |
| `kohakumoss-doctor` | `--json=""`, `--memory=False`, `--clear-failures=False` | Prints a report; a JSON path is overwritten, without creating its parent directory; clear-failures mutates recorded backend failures |
| `kohakumoss-tune` | `--device=cuda`, `--dtype=bf16` (or fp16), `--sizes 1024x1024` (one or more HxW), `--variant=base`, `--text-tokens=512`, `--batch=4`, `--backward=False`, `--trainable=adapter` (or full) | Runs device kernels and timings and writes the per-card tuning table; not a read-only environment check |

Source: [doctor.main/report](../../src/kohakumoss/cli/doctor.py), [tune.main/tune](../../src/kohakumoss/cli/tune.py). Doctor returns fields `os/os_release/python/torch_cuda/packages/devices/missing/msvc/compilers/kernels/backend_status/backend_failures`; `--memory` adds `memory_plans`. Missing requirements or recorded failures do not automatically make its exit status nonzero: inspect the fields. Partial path redaction is not a guarantee that all report content is safe to publish. The Python `tune()` returns the number of problems run; the CLI does not use that number as its exit status. Exceptions propagate.

### kogine scripts and key defaults

Knobs are module-level uppercase names, overridden with repeated `--set KEY=VALUE`. Values must be coerced against the script schema; a string is not already a validated boolean or number. See the [script catalogue](scripts.md) for the full inventory.

| Entry | Integration knobs (script defaults, not recipe overrides) |
| --- | --- |
| [train/moss.py](../../scripts/train/moss.py) `main` | `WEIGHTS=""`, `DATASETS=[]` need configuration; `ALGO="lora"`, `LR=1e-4`, `STEPS=1000`, `PRECISION="bf16"`, `MEMORY_PROFILE="balanced"`, `DRIVER="native"`, `PARALLEL="none"`, `DEVICES=(0,)` |
| [sample/sample.py](../../scripts/sample/sample.py) `main/load_items` | `WEIGHTS=""`, `PROMPT=""`, `PROMPTS=[]`, `PROMPTS_FILE=""`, `OUT=""`; `DEVICE="cuda"`, `STEPS=50`, `CFG=3.0`, `SHIFT=5.0`, `BATCH=4`, `SIZES=[(1024,1024)]`, `SEEDS=[]`, `SEED_BASE=0`, `TUNE="off"` |
| [export/export.py](../../scripts/export/export.py) `main` | `WEIGHTS/TRANSFORMER/ADAPTER/OUT=""`, `VARIANT="base"`, `FORMAT="comfy"`, `STYLE="lora_unet"`, `SAVE_DTYPE="bf16"`, `MULTIPLIER=1.0`, `FUSED_QKV="auto"`, `DEVICE="cpu"`; formats: [export guide](export-to-comfy.md) |

Sampling assembles `PROMPTS`, nonempty `PROMPT`, then file entries. A file is text (one nonblank prompt per line) or a JSON list of strings or `{prompt, negative, size: [H,W], seed}` objects. Missing sizes use `SIZES`, repeating its last entry; missing seeds use `SEEDS[i]`, then `SEED_BASE+i`. No prompts raises `SystemExit`. Outputs are `<index>_<seed>.png` and `metadata-<first index>.json`, whose top-level fields are `knobs/settings/run/images`. Numbering continues after existing outputs, without a multi-process allocation lock: do not share OUT across simultaneous sampling processes.

Primary data-preparation scripts are `scripts/data/inspect_data.py`, `cache_text.py` and `cache_latents.py`; catalog building and cache merging/cleanup are listed in [datasets](datasets.md) and the script catalogue. They can scan, encode or write caches; they are not schema-only queries.

## 2. Run and file contract

Sources: [training.entry.run/_start/worker](../../src/kohakumoss/training/entry.py), [training.knobs.run_name/check](../../src/kohakumoss/training/knobs.py), [logging.sinks](../../src/kohakumoss/logging/sinks.py).

| Knob | Default | Meaning |
| --- | --- | --- |
| `OUTPUT_DIR` / `RUN_NAME` | `"runs/moss"` / `""` | `Path(OUTPUT_DIR) / (RUN_NAME or "run")`; no automatically unique run ID |
| `LOGGER` / `PROJECT` | `"console"` / `"kohakumoss"` | Built-ins: console and wandb; both write metrics.jsonl; wandb needs its package and may access the network |
| `LOG_EVERY` / `METRICS_EVERY` | `10` / `None` | Console / metric interval; None uses LOG_EVERY; metrics also fire on the first executed step (resume step+1) and the last step |
| `RESUME` | `""` | No restore; `"latest"` selects this run's newest complete checkpoint; another string is a path |
| `SAVE_EVERY` / `SAVE_MINUTES` | `500` / `0` | Every N steps / minutes; 0 disables; minute condition is checked every LOG_EVERY steps |
| `KEEP_LAST` / `FINAL_CHECKPOINT` | `3` / `"auto"` | Keep newest complete checkpoints (all if KEEP_LAST<=0); auto writes a final checkpoint on normal completion when periodic saves are enabled |
| `SAVE_FORMAT` / `SAVE_MERGED` | `"comfy"` / `False` | Normal-completion exports: comfy/hf/both/none; not a substitute for a resume checkpoint |
| `CHECKPOINT_EXPORT` / `SAVE_FUSED_QKV` | `True` / `"auto"` | Export at checkpoints too; auto follows the DiT file's QKV layout |

RUN_NAME goes straight into Path composition, without a directory sandbox. A UI should constrain path components and fix cwd or use absolute OUTPUT_DIR. Do not run independent jobs against one output directory: there is no run ownership lock. Rank 0 writes the config/global log; other ranks get NullLogger. Diagnostic and per-rank files have separate writing paths.

| Path inside run | Content and update behavior |
| --- | --- |
| `config.json` | Effective knob snapshot directly overwritten by rank 0 on startup; UTF-8 JSON, indent=2, otherwise non-JSON values stringify; not hot-reload input |
| `log.txt` | Text log; messages are not a fixed state protocol |
| `metrics.jsonl` | UTF-8 append, one flushed JSON row at a time; see below |
| `checkpoints/step_XXXXXXXX/` | Resume state; first written under the matching `.partial/` directory, then renamed |
| `checkpoints/step_XXXXXXXX/state.json` | Checkpoint, not live state: `step/loader/tokens/samples/joint_sq/text_tokens` plus layout-specific fields |
| `instability/step_XXXXXXXX.json` | Non-finite / loss-spike diagnostic: `step/reason/loss/loss_avg/grad_norm/loss_scale/lr/batch_ids/metrics/nonfinite_grads`; direct write |
| `samples/`, `export/`, `ranks/` | Optional sampling, export and per-rank diagnostics; their existence does not prove success |

Without a resolved resume checkpoint, `fresh_run()` rotates only nonempty metrics.jsonl and log.txt to the first available `metrics.N.jsonl` / `log.N.txt`. It does not clear checkpoints, samples or exports. **`RESUME="latest"` with no complete checkpoint returns None and starts fresh, not an error.** Launchers requiring restore should check first.

[CheckpointManager](../../src/kohakumoss/training/checkpoint/manager.py) lists only `step_\d{8}` directories containing state.json and trainable.safetensors. This is an existence check, not a checksum. Each rank writes `rng.<rank>.pt` and optionally `optim.<rank>.pt`; rank 0 writes trainable.safetensors, optional ema.safetensors, optim.pt, then state.json last. Rank synchronization surrounds the commit. An existing directory at the same step is replaced; KEEP_LAST prunes old checkpoints. Checkpoint `export/` is written **after** directory commit, so directory appearance does not guarantee export completion. See [checkpoint.io](../../src/kohakumoss/training/checkpoint/io.py) and [Parts](../../src/kohakumoss/training/drivers/common.py).

Explicit resume paths bypass latest's completeness filtering. Missing/corrupt files can raise I/O, JSON or tensor-loading errors; missing RNG for the current rank raises FileNotFoundError. UIs should not deserialize `.pt` files just to display status or modify checkpoint content.

## 3. Reading metrics.jsonl

Writers: [ConsoleLogger.log/resume_at](../../src/kohakumoss/logging/sinks.py), [LogMetrics/Throughput](../../src/kohakumoss/training/callbacks/monitor.py).

| Field | Type / units / availability |
| --- | --- |
| `step`, `time` | Integer step and float Unix seconds; present in normal built-in training rows |
| `loss`, `grad_norm`, `lr`, `loss_scale` | Training scalars, potentially non-finite |
| `skipped_steps`, `nonfinite_steps`, `nonfinite_params` | JSON numbers (converted to float); first two cumulative, last is the count of parameters in the current step, not cumulative |
| `tokens_per_s`, `samples_per_s`, `step_time`, `tflops` | Tokens/s, samples/s, seconds/step, TFLOP/s; `mfu` only with a known device peak, a fraction rather than percent |
| `vram_*`, `te_*`, `vae_*`, `xfer_*`, etc. | Optional by hardware/callback; VRAM is GiB; see [logging](logging.md) |

Illustrative row, not a measurement from this documentation check:

```json
{"step":1,"time":1700000000.0,"loss":0.25,"grad_norm":0.7,"lr":0.0001,"loss_scale":1.0,"skipped_steps":0.0,"nonfinite_steps":0.0,"nonfinite_params":0.0}
```

There is no schema_version, heartbeat, completion event or row-per-step guarantee. Default `json.dumps` allows NaN/Infinity/-Infinity; strict JSON consumers need an explicit non-finite policy, not blanket text replacements. Preserve/ignore extension fields and do not require identical key sets in every row.

**Consumer guidance, not additional writer guarantees:** retain a byte offset and incomplete tail; parse only complete newline-delimited rows. Detect replacement, truncation and temporary absence, then reopen. Flush is not fsync or a multi-process transaction. Startup config.json and diagnostic JSON are direct writes: retry incomplete reads.

At resume step S, `resume_at(S)` closes the stream, writes rows with step>S to a fresh metrics.N.jsonl, directly rewrites the original with the retained rows, then reopens append. Without later rows it need not rewrite. Invalid step parsing falls back to -1 and is retained: malformed rows are not guaranteed to be cleaned. Resume/rotation can invalidate the old descriptor and offset.

## 4. Lifecycle and failures

- The current training boundary is process launch, artifacts and exit. There is no HTTP service, pause/resume RPC, status.json heartbeat or knob-file hot reload. config.json and checkpoint state.json are not control channels.
- `training.entry.run(g)` collects/normalizes and checks a full knob set before building data/model and training; problems become one `SystemExit("invalid configuration:...")`. It does not fill every script default into an arbitrary sparse dict. Normal process completion is 0; uncaught exceptions fail. Outer kogine exception handling depends on the installed version.
- [shutdown.install](../../src/kohakumoss/training/shutdown.py) installs once, only from the main thread. A Python-handled SIGTERM raises `SystemExit(128+signum)`, normally 143. Exit cleanup kills multiprocessing children and the forkserver. Windows TerminateProcess, SIGKILL and other forced termination do not guarantee Python cleanup.
- SIGTERM is exception-style shutdown, **not save-and-stop**. Default [NativeDriver.run](../../src/kohakumoss/training/drivers/native.py) finishes its final checkpoint after the normal loop; [worker](../../src/kohakumoss/training/entry.py) exports on the normal-return path. Existing finally blocks attempt cleanup, but interruption need not create a fresh checkpoint. Restore from the last complete checkpoint.
- Distributed entry may spawn rank processes; see [multi-GPU](multi-gpu.md) for platform/layout restrictions. Manage the whole job, not just a single PID. No new log row does not imply death: building, caching or tuning may still be in progress. Use process status/exit for lifecycle, and files for recoverable artifacts.

## 5. Reusable Python entries

These are existing entry points, not a cross-version compatibility promise for every internal class. Prefer subprocess isolation for UIs; do not import the training stack, probe devices or execute configs on the GUI thread.

| Module / symbol | Input → result and boundaries |
| --- | --- |
| [cli.config.validate](../../src/kohakumoss/cli/config/validate.py) `config_knobs(config: Path, root=None)` | `(script: str, knobs: dict)`; first scripts/...py in the config docstring selects the script; layers get their base prepended; no target or sweep/generator raises ValueError |
| Same module `compose(root, script, layers, own)` / `config_problems(config, root=None)` | Defaults → ordered layers → own; dict / list[str]. `problems()` returns [] for scripts outside train/sample/data: **not a universal script validator**. Reading/executing configs may still raise |
| [common.script_schema](../../src/kohakumoss_common/script_schema.py) `static_schema(path: str)` | Does not import the target script; returns `(ScriptSchema, opaque_names)`, or None for unsupported statements. Unknown values belong to the opaque set; do not invent defaults. This helper itself depends on kohakuengine |
| [training.entry](../../src/kohakumoss/training/entry.py) `run(g: dict)` | Full script globals → [TrainState](../../src/kohakumoss/training/state.py) in none/threaded mode, None for process-group launch; mutates signals, process tuning state, devices and files, not a concurrent side-effect-free call |
| [kohakumoss_data](../../src/kohakumoss_data/__init__.py) `DataConfig.from_knobs(k)` / `data_problems(cfg)` / `build_loader(cfg, rank=0, world=1, transform=None, writer=True)` | DataConfig / problem list / Loader. from_knobs selects fields and resolves cache_dir, but does not call data_problems. Loader construction may scan/write catalogs; designate one writer per run, other ranks False |
| [common.registry](../../src/kohakumoss_common/registry.py) `Registry(family)` / `build(choice, registry, key="name", **defaults)` | Registry name/import path/dict/class/buildable/instance → built result; None → None; dict arguments override defaults. Unknown simple names raise KeyError; import/constructor errors propagate. Trusted extensions only |

An empty DataConfig CACHE_DIR resolves to `cache/` under the first dataset with a path when caches are enabled; file/glob paths use the parent directory. It can stay empty when both tensor caches are off; an explicit cache_dir is retained. See [data.config](../../src/kohakumoss_data/config.py) and [data](../concepts/data.md); dataclass defaults are not necessarily identical to script defaults.

### MossPipeline

Public import: `from kohakumoss.inference import MossPipeline, GenerateResult`. Source: [inference.pipeline](../../src/kohakumoss/inference/pipeline.py); full usage: [inference](inference.md).

- `MossPipeline.from_pretrained(path, ...) -> MossPipeline`: local ComfyUI models root, pipeline directory or base/sft weights root. Key defaults: `variant="base"`, `dtype=torch.bfloat16`, `device="cuda"`, `adapters=()`, `adapter_mode="merge"`, `fp16_fix="none"`, `tune="off"`, `text_encoder=True`, `vae=True`. Component overrides: transformer, text_encoder_file, vae_file; planning options are in the existing guide. Loads models and changes process tuning state; no thread-safety guarantee.
- `generate(prompts=None, negative_prompts=None, sizes=(1024,1024), steps=50, cfg=3.0, shift=5.0, skip_layers=None, seeds=None, batch=4, cfg_mode="packed", output="pil", prompt_embeds=None, negative_embeds=None, step_callback=None, progress=True)`: sizes are **(H,W)**, one image per prompt, default seeds range(n), default skip_layers from the architecture. Supply nonempty prompts or embeddings and consistent per-item sizes/seeds/negative conditions. This Python method is not the CLI knob validator; loading/shape/index errors propagate without a uniform error object.
- Returns `GenerateResult(images, latents, settings)`. Output pil gives PIL images, tensor gives `[1,3,H,W]` tensors in [-1,1], latent skips decoding and puts latents into images too; unknown output raises KeyError. Settings include call parameters, model sources and statistics. No image files are written by generate itself.
- `from_components(backbone, text_encoder=None, vae=None, device=None, memory_plan=None, weight_stream=None)` shares existing components without reloading or taking over the trainer's TUNE. `encode(prompts)` returns embeddings; `decode(latents)` returns pixel tensors.

Minimal config reader (trusted config only; not executed during this check):

```python
from pathlib import Path
from kohakumoss.cli.config.validate import config_knobs, problems

script, knobs = config_knobs(Path("configs/user/my-lora.py"))
errors = problems(script, knobs)
if errors:
    raise ValueError("\n".join(errors))
print(script, knobs["OUTPUT_DIR"], knobs["RUN_NAME"])
```

Minimal sampling call (installed dependencies, an available device and local weights required; no automatic weight preparation):

```python
from kohakumoss.inference import MossPipeline

pipe = MossPipeline.from_pretrained("ComfyUI/models", tune="off")
result = pipe.generate(["a lighthouse at dusk"], seeds=[0], progress=False)
result.images[0].save("sample.png")
```

Training uses a generated, validated config with its required caches prepared. STEPS is the total target, not extra resumed steps:

```bash
kohakumoss-config check configs/user/my-lora.py
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set RUN_NAME=my-lora --set STEPS=1000
# Restore the same run after confirming a complete checkpoint exists.
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set RUN_NAME=my-lora --set RESUME=latest --set STEPS=2000
```

Verification scope: static comparison with parsers, script defaults, file write/restore paths, public symbols and local links. No training, inference, GPU probing/tuning or config execution was performed. Example parameters were checked against source; they are not a record of a successful hardware run.
