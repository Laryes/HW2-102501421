# 第二次挑战：SGLang 前缀缓存实验与请求流程阅读

本目录保存第二次挑战的实验脚本、实验备份和源码阅读材料。任务一比较共享前缀与分散前缀负载的缓存效果；任务二沿请求接收、等待队列、前缀匹配、Prefill/Decode、缓存更新和流式输出梳理 SGLang 请求主流程。

实验结果来自真实运行，已经整理在 `results/target1/`，可以直接查看，无需先解压备份。实验脚本位于 `src/target1/prefix_benchmark.py`。下文命令均从提交根目录执行，GPU 服务和实验脚本在云端 Linux 环境运行。

## 1. 文件与目录

```text
HW2-姓名/
├── README.md
├── report.pdf                            # 由 report.docx 导出
├── AI使用说明情况（第二次挑战）.pdf      # 由同名 Word 文档导出
├── src/
│   └── target1/
│       └── prefix_benchmark.py
└── results/
    └── target1/
        ├── config.json
        ├── server_info.json
        ├── nvidia-smi.txt
        ├── prefix.json
        ├── service_warmup.json
        ├── comparison.csv
        ├── validation.json
        ├── benchmark.log
        ├── server.log
        ├── dispersed_prefix/
        │   ├── inputs.json
        │   ├── flush.json
        │   ├── request-01.json ... request-32.json
        │   ├── requests.csv
        │   └── summary.json
        └── shared_prefix/
            ├── inputs.json
            ├── flush.json
            ├── warmup.json
            ├── request-01.json ... request-32.json
            ├── requests.csv
            └── summary.json
```


## 2. 实验环境

实验在 AutoDL Linux 云端 GPU 实例中运行，本机 Windows 用于下载文件、阅读源码和整理报告。

| 项目 | 本次记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，`nvidia-smi` 显示总显存 24564 MiB |
| NVIDIA 驱动 | 580.76.05 |
| Python | 3.10.8 |
| SGLang | 0.5.14 |
| Ray | 2.56.0 |
| PyTorch | `config.json` 记录为 2.11.0；此前环境检查显示 CUDA 13 构建 |
| aiohttp | 3.14.3 |
| Transformers | 5.8.1 |
| 模型 | Qwen/Qwen3-0.6B |
| Attention backend | flashinfer |
| 上下文长度 | 4096 |
| 静态显存比例 | 0.5 |
| Radix Cache | 开启，`disable_radix_cache=false` |
| Chunked prefill size | 2048 |
| Page size | 1 |
| 流式输出间隔 | 1 |
| Overlap 调度 | 开启，`disable_overlap_schedule=false` |

软件版本以 `config.json` 为依据，服务配置以 `server_info.json` 为依据，GPU 与驱动以 `nvidia-smi.txt` 为依据。`nvidia-smi` 中的 CUDA Version 表示驱动支持能力，不能单独用于确认编译工具包版本。本次使用的数据盘 CUDA 工具包路径为 `/root/autodl-tmp/cuda13`。

## 3. 安装与环境准备

以下命令在云端 Linux 终端执行。已保留原实例环境时，直接激活虚拟环境即可，不必重新安装。

```bash
source /root/autodl-tmp/first-challenge/.venv/bin/activate
python -c "import importlib.metadata as m; print('SGLang:', m.version('sglang')); print('Ray:', m.version('ray'))"
```

新环境的依赖安装参考如下。运行机器需要 NVIDIA GPU、可用驱动，以及与 PyTorch/FlashInfer 相容的 CUDA 编译工具和运行库；仅安装 Python 包不一定能满足首次内核编译要求。

```bash
mkdir -p /root/autodl-tmp/first-challenge
python -m venv /root/autodl-tmp/first-challenge/.venv
source /root/autodl-tmp/first-challenge/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "sglang==0.5.14" "ray==2.56.0" \
  "torch==2.11.0" "aiohttp==3.14.3" "transformers==5.8.1"
```

CUDA 工具包方面，本次原镜像的 CUDA 11.8 编译器未能满足依赖要求，随后通过 Conda 安装 CUDA 13 工具包：

```bash
conda create -y \
  -p /root/autodl-tmp/cuda13 \
  -c nvidia/label/cuda-13.0.0 \
  cuda-toolkit
```

上述 Python 安装命令未锁定所有间接依赖，也未单独指定 PyTorch wheel 的 CUDA 构建；重新搭建时需检查 `torch.version.cuda` 和 `nvcc --version`，不能保证在任意基础镜像上直接复现完全相同的环境。

模型在第一次挑战中已下载到 `/root/autodl-tmp/models/Qwen3-0.6B`。新实例可下载同一模型到该路径：

```bash
python -m pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir="/root/autodl-tmp/models/Qwen3-0.6B"
)
PY
```

已有本地模型时无需重复下载。模型文件需包括权重、配置和 tokenizer 文件；脚本使用 `local_files_only=True` 加载 tokenizer。

## 4. 启动模型服务

将提交文件夹上传到云端 `/root/autodl-tmp/second-challenge/`，确保该目录下已有 `src/target1/prefix_benchmark.py`。在第一个终端执行以下命令，并保持该终端运行。本次使用的模型和虚拟环境路径如下。

```bash
source /root/autodl-tmp/first-challenge/.venv/bin/activate

export CUDA_HOME=/root/autodl-tmp/cuda13
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /root/autodl-tmp/second-challenge/replay-results
cd /root/autodl-tmp/second-challenge

python -m sglang.launch_server \
  --model-path /root/autodl-tmp/models/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 30000 \
  --context-length 4096 \
  --mem-fraction-static 0.5 \
  2>&1 | tee replay-results/server.log
```

未传入关闭 Radix Cache 的参数。本次实际配置启用了 overlap 调度。任务二按要求阅读 `event_loop_normal` 作为逻辑主线，并不意味着实验服务实际采用普通非 overlap 循环。

服务就绪后，在第二个终端检查：

```bash
curl --max-time 10 -sS http://127.0.0.1:30000/v1/models
```

返回包含 `Qwen/Qwen3-0.6B` 的模型列表表示接口可访问。正式实验脚本还会执行短流式请求预热并检查生成结果。

## 5. 运行任务一实验

在已上传的提交根目录运行 `src/target1/prefix_benchmark.py`。复跑结果写入独立的 `replay-results/`，保留已提交的原始测量数据。服务就绪后，在第二个终端执行：

```bash
source /root/autodl-tmp/first-challenge/.venv/bin/activate
cd /root/autodl-tmp/second-challenge
set -o pipefail
mkdir -p replay-results

python src/target1/prefix_benchmark.py \
  --base http://127.0.0.1:30000 \
  --model /root/autodl-tmp/models/Qwen3-0.6B \
  --results replay-results \
  2>&1 | tee replay-results/benchmark.log
```

`set -o pipefail` 使脚本失败时管道保留失败状态。每次执行创建新的 `replay-results/run-<UTC时间戳>/`，两组数据位于该目录的 `target1/<组名>/`，配置与对照表位于 run 根目录。这是脚本自动生成的结构；本次提交将已有运行的公共文件整理到 `results/target1/`，将组文件整理到其下的两组目录，文件内容未改动。`replay-results/benchmark.log` 与 `server.log` 记录复跑日志；再次复跑前可备份这两个日志。

### 负载与执行顺序

- 原生 `POST /generate`，传入 `input_ids`，开启 `stream=true`。
- 每组 32 条测量请求，客户端最大并发为 8。
- 每条输入 2112 个 token，输出固定为 16 个 token。
- 采样参数为 `temperature=0`、`max_new_tokens=16`、`ignore_eos=true`、`sampling_seed=2026`。
- 共享组使用 2048-token 公共前缀和 64-token 独立后缀，各后缀首 token 不同。
- 分散组复制对应共享请求，仅将第一个 token 改为各请求互不相同的 token，阻止共同前缀匹配；输入长度、后缀、请求编号与提交顺序保持一致。
- 合成负载用于测量缓存效果，不用于评价回答质量。
- 脚本先发送短请求进行服务预热，再测分散组，最后测共享组。
- 每组测量前调用 `POST /flush_cache?timeout=30`，验证 HTTP 200 和成功响应；等待本脚本此前请求结束后才开始下一组。
- 共享组清缓存后先发送一条仅含 2048-token 公共前缀的预热请求，等待完成后再测量。服务预热和前缀预热均不计入测量结果。
- 请求通过信号量限制并发；获得信号量后的提交时刻开始记录单请求延迟。服务端批次大小由调度器决定，不等于固定 8。
- 测量期间不要从其他客户端发送请求，以免污染缓存和资源竞争条件。

脚本检查最终元数据、输入/输出长度、流式 token 计数和完成标记。如果流式消息合并多个新 token，脚本会将请求标记失败，避免把多个 token 的接收时刻误当作精确首 token 时刻。

### 指标定义

| 指标 | 本脚本口径 |
|---|---|
| 成功率 | 成功测量请求数 / 32；成功要求 HTTP 200、完整流式结束和长度检查通过 |
| 请求吞吐量 | 成功测量请求数 / 本组测量墙钟时间，单位 requests/s |
| 输出吞吐量 | 成功请求输出 token 总数 / 本组测量墙钟时间，单位 tokens/s |
| 缓存命中率 | `sum(cached_tokens) / sum(prompt_tokens)` |
| 实际 Prefill token 数 | 按题目口径计算 `sum(prompt_tokens - cached_tokens)` |
| TTFT | 请求提交至客户端收到首个生成 token 的时间 |
| TPOT | `(收到最后一个 token 的时间 - 收到首个 token 的时间) / (输出 token 数 - 1)`，本实验分母为 15 |
| 端到端延迟 | 请求提交至客户端收到 SSE `[DONE]` 的时间 |
| p50、p95 | 对逐请求指标采用线性插值分位数 |

延迟为客户端测量值，包含 HTTP、服务端排队及输出传输等开销，不是纯 GPU 内核耗时。客户端信号量等待不计入单请求延迟，但在组墙钟时间内。原始事件先保存在内存中，组计时结束后再写入文件和打印，避免文件/终端输出干扰测量。汇总指标使用成功请求；存在失败时成功率会明确显示，且完成标准检查不会通过。

## 6. 本次实验结果及报告对应关系

本次提交数据位于 `results/target1/`。原始运行编号为 `run-20261003T024733.130495Z`，对应北京时间 2026-10-03 10:47:33；该编号用于追溯运行，当前提交目录不再嵌套这一层。

| 指标 | 分散前缀 | 共享前缀 |
|---|---:|---:|
| 成功请求数 | 32/32 | 32/32 |
| 成功率 | 100% | 100% |
| 输入 token 总数 | 67,584 | 67,584 |
| 缓存 token 总数 | 0 | 65,536 |
| 缓存命中率 | 0% | 96.97% |
| 实际 Prefill token 总数 | 67,584 | 2,048 |
| 请求吞吐量（requests/s） | 29.95 | 105.49 |
| 输出吞吐量（tokens/s） | 479.12 | 1687.89 |
| TTFT p50 / p95（ms） | 141.20 / 198.40 | 32.06 / 39.85 |
| TPOT p50 / p95（ms） | 8.48 / 13.63 | 2.60 / 3.44 |
| 端到端延迟 p50 / p95（ms） | 262.03 / 277.75 | 72.58 / 80.81 |

`validation.json` 中 `assignment_checks_passed=true`，对应两组均成功 32 条、共享组命中率更高且 Prefill token 更少。脚本成功判定还逐条检查输入与输出长度。

共享组每条请求命中 2048 个 token，剩余 64 个需要计算，所以累计缓存量为 `32×2048=65536`，累计 Prefill 量为 `32×64=2048`。分散组每条总共需要计算 2112 个输入 token，可按服务配置分块执行。

以上为一次运行的数据。TTFT、TPOT 与吞吐量受硬件、调度和当前负载影响。共享组 TPOT 的改善不能直接解释为前缀复用消除了 Decode；减少 Prefill 后的资源竞争和调度变化可能影响 TPOT，需在报告中区分机制与推断。

| 报告内容 | 对应提交文件（相对于 `results/target1/`） |
|---|---|
| 环境、软件版本和固定实验参数 | `config.json`、`nvidia-smi.txt` |
| 服务启动与实际运行配置 | `server_info.json`、`server.log`；客户端运行输出见 `benchmark.log` |
| 两组对照主表 | `comparison.csv`；两组 `<组名>/summary.json` |
| 逐请求长度、缓存量与延迟 | 两组 `<组名>/requests.csv` |
| 流式接收时刻、输出编号与元数据 | 两组 `<组名>/request-01.json` 至 `request-32.json` |
| 实际使用的输入 token 编号 | 两组 `<组名>/inputs.json`；公共前缀见 `prefix.json` |
| 清缓存成功证据 | 两组 `<组名>/flush.json` |
| 服务预热与共享前缀预热 | `service_warmup.json`、`shared_prefix/warmup.json` |
| 完成标准检查 | `validation.json` |

## 7. 任务二源码定位

任务二的流程图与说明见报告。源码阅读材料包括 `scheduler.py`、`source_index.txt`、`source_stage1.txt` 至 `source_stage3.txt` 和 `source_reading.tar.gz`，本地留存用于阅读核对；这些文件不参与任务一实验运行，也不需要上传模型权重或虚拟环境。源码摘录中的原始行号用于导航，索引包含其他模型和后端的同名函数，阅读时需筛选普通生成请求路径。

任务二的主要阅读位置如下，路径均以 `sglang/srt/` 为起点。行号来自本次云端源码副本，不保证其他安装副本行号一致。

| 环节 | 文件与关键函数 |
|---|---|
| 请求接收与发送 | `entrypoints/http_server.py:769` 的 `generate_request`；`managers/tokenizer_manager.py:576` 的 `generate_request`、`:1319` 的 `_send_one_request` |
| 普通调度循环与入队 | `managers/scheduler.py:1505` 的 `event_loop_normal`、`:1998` 的 `handle_generate_request`、`:2258` 的 `_add_request_to_queue`；入队语句位于 `:2265` |
| 构建 Prefill 批次 | `managers/scheduler.py:2702` 的 `get_new_batch_prefill`；`managers/schedule_policy.py:85` 的 `match_prefix_for_req`、`:858` 的 `PrefillAdder.add_one_req` |
| 前缀长度与新输入准备 | `managers/schedule_batch.py:1123` 的 `Req.init_next_round_input`、`:2008` 的 `ScheduleBatch.prepare_for_extend` |
| 缓存匹配和登记 | `mem_cache/radix_cache.py:358` 的 `match_prefix`、`:438` 的 `cache_finished_req`、`:485` 的 `cache_unfinished_req` |
| 模型执行 | `managers/scheduler.py:3145` 的 `run_batch`；`managers/tp_worker.py:482` 的 `forward_batch_generation`；`model_executor/model_runner.py:2896` 的 `forward` |
| 结果处理与清理 | `managers/scheduler_components/batch_result_processor.py` 的 `process_batch_result_prefill`、`process_batch_result_decode`；`mem_cache/common.py` 的 `release_kv_cache` |
| 流式结果返回 | `managers/scheduler_components/output_streamer.py` 的 `stream_output`；`managers/detokenizer_manager.py`；`managers/tokenizer_manager.py` 的 `handle_loop`、`_handle_batch_output`、`_wait_one_response` |

阅读重点是各函数的输入、主要动作、输出去向。前缀匹配函数有多个调用入口，不能把所有匹配函数写成必然依次调用。缓存登记保存 token 序列与已计算 KV 存储索引的映射；结果流式返回与模型逐 token 生成是不同层面的行为。

