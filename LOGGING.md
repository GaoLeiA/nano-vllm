# 详细日志功能使用指南

## 概述

nano-vllm 现已添加全面的日志系统，可以追踪整个推理流程的详细信息，包括：

- **输入输出形状** - 各个阶段的张量维度
- **内存占用** - GPU内存分配和使用情况
- **通信细节** - 张量并行的通信操作（如果启用）
- **调度决策** - 批次组成和序列调度
- **块管理** - KV cache块的分配和缓存命中
- **执行时间** - 关键操作的耗时

### 🎯 智能日志频率控制

为了避免DECODE阶段产生过多重复日志，系统采用了智能频率控制：

**INFO级别（默认）**：
- PREFILL批次：每次都记录（较少发生）
- DECODE批次：仅在关键时刻记录
  - ✅ 第一个decode step
  - ✅ 每10个step
  - ✅ 批次大小改变时
  - ✅ 发生序列抢占时
- ModelRunner的详细信息（张量形状等）自动降级到DEBUG

**DEBUG级别**：
- 记录所有step的详细信息
- 包括所有张量形状和中间结果

这样可以保持日志简洁易读，同时在需要时仍能看到完整细节。

## 日志级别控制

### 方式1：环境变量（推荐）

```bash
# Windows PowerShell
$env:NANOVLLM_LOG_LEVEL="DEBUG"
python example.py

# Linux/Mac
export NANOVLLM_LOG_LEVEL=DEBUG
python example.py
```

### 方式2：在代码中设置

```python
import os
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

# 然后导入 nanovllm
from nanovllm import LLM, SamplingParams
```

### 可用的日志级别

- **`DEBUG`** - 最详细，显示所有日志（包括张量形状、每个step的细节）
- **`INFO`** - 标准详细程度，显示主要流程和统计信息
- **`WARNING`** - 仅显示警告和错误
- **`ERROR`** - 仅显示错误

## 日志内容说明

### 1. 初始化阶段

```
[INFO] ================================================================================
[INFO] Initializing LLMEngine
[INFO] Model: ~/huggingface/Qwen3-0.6B/
[INFO] Configuration: tensor_parallel_size=1, max_model_len=2048, ...
[INFO] Initializing ModelRunner (rank 0)
[INFO] Loading model: ~/huggingface/Qwen3-0.6B/
[INFO] Model config: num_layers=28, hidden_size=896, ...
[INFO] After model loading - GPU Memory: Allocated=X.XXG B, Reserved=X.XXGB
[INFO] Allocating KV cache
[INFO] KV cache configuration: num_kv_heads=2, head_dim=128, block_size=16
[INFO] Allocating 512 KV cache blocks, total_size=X.XXGB
[INFO] After KV cache allocation - GPU Memory: Allocated=X.XXGB
```

**关键信息**：
- 模型配置参数
- 加载后的GPU内存占用
- KV cache的配置和大小
- 分配后的GPU内存占用

### 2. 生成阶段

```
[INFO] --------------------------------------------------------------------------------
[INFO] Starting generation for 2 prompts
[INFO] Total prompt tokens: 156
```

**关键信息**：
- 批次中的prompt数量
- 总的输入token数

### 3. Prefill阶段（首次处理提示词）

```
[INFO] Scheduled PREFILL batch: size=2, batched_tokens=156, 
      seqs=[seq_0(len=78,cached=0), seq_1(len=78,cached=0)]
[DEBUG] [Rank 0] Preparing PREFILL batch: 2 sequences
[DEBUG] [Rank 0] PREFILL tensors: input_ids=[156], positions=[156], slot_mapping=[156]
[DEBUG] [Rank 0] PREFILL: max_seqlen_q=78, max_seqlen_k=78, total_tokens=156
[DEBUG] [START] [Rank 0] Model forward PREFILL
[DEBUG] [END] [Rank 0] Model forward PREFILL - Elapsed: XX.XXms
[DEBUG] [Rank 0] Output: logits=[156, 151936]
```

**关键信息**：
- Prefill批次大小和序列信息
- 输入张量的形状
- 序列的最大长度
- 模型前向传播的耗时
- 输出logits的形状

### 4. Decode阶段（逐token生成）

**INFO级别（简洁模式）**：
```
[INFO] Scheduled DECODE batch (step 1): size=2, preempted=0, seqs=[seq_0(len=79), seq_1(len=79)]
... (步骤2-9：仅DEBUG级别记录)
[INFO] Scheduled DECODE batch (step 10): size=2, preempted=0, seqs=[seq_0(len=88), seq_1(len=88)]
... (步骤11-19：仅DEBUG级别记录)
[INFO] Scheduled DECODE batch (step 20): size=2, preempted=0, seqs=[seq_0(len=98), seq_1(len=98)]
```

**DEBUG级别（详细模式）**：
```
[INFO] Scheduled DECODE batch (step 1): size=2, preempted=0, seqs=[seq_0(len=79), seq_1(len=79)]
[DEBUG] [Rank 0] Preparing DECODE batch: 2 sequences
[DEBUG] [Rank 0] DECODE tensors: input_ids=[2], positions=[2], context_lens=[2], block_tables=[2, 5]
[DEBUG] [START] [Rank 0] Model forward DECODE
[DEBUG] Decode attention: k_cache=[2, 28, 512, 16, 2, 128], v_cache=[2, 28, 512, 16, 2, 128]
[DEBUG] [END] [Rank 0] Model forward DECODE - Elapsed: XX.XXms
[DEBUG] [Rank 0] Output: logits=[2, 151936]
[DEBUG] Scheduled DECODE batch (step 2): size=2, preempted=0, seqs=[seq_0(len=80), seq_1(len=80)]
... (每个step都有详细日志)
```

**关键信息**：
- Decode批次信息（周期性记录）
- 输入张量形状（DEBUG级别）
- Context长度（DEBUG级别）
- KV cache的完整形状（DEBUG级别）
- Decode的耗时（DEBUG级别）

### 5. 块管理

```
[DEBUG] Allocating blocks for seq_id=0, num_blocks=5
[DEBUG] Allocated blocks for seq_id=0: new=5, cache_hits=0, 
        free_blocks=507, used_blocks=5
[DEBUG] Deallocated blocks for seq_id=0: freed=5/5, free_blocks=512
```

**关键信息**：
- 为每个序列分配的块数
- 新分配的块和缓存命中的块
- 空闲和已使用的块数量

### 6. Attention层

```
[DEBUG] Attention forward: is_prefill=True, q=[156, 16, 64], k=[156, 2, 64], v=[156, 2, 64]
[DEBUG] Storing KV cache: N=156, num_heads=2, head_dim=64, num_slots=156
[DEBUG] Attention output: [156, 16, 64]
```

**关键信息**：
- 是Prefill还是Decode模式
- Query、Key、Value的形状
- 存储到KV cache的数量
- Attention输出的形状

### 7. 序列完成

```
[DEBUG] Sequence finished: seq_id=0, output_len=256, reason=max_tokens
[DEBUG] Postprocess: 1/2 sequences finished
```

**关键信息**：
- 完成的序列ID
- 生成的token数量
- 完成原因（EOS或达到max_tokens）

### 8. 最终统计

```
[INFO] Generation complete: 42 steps, total_output_tokens=512, 
       prefill_time=0.15s, decode_time=2.43s
[INFO] After generation - GPU Memory: Allocated=X.XXGB
[INFO] --------------------------------------------------------------------------------
```

**关键信息**：
- 总步数
- 生成的总token数
- Prefill和Decode的总耗时
- 最终GPU内存占用

## 示例输出

运行 `example.py` 的完整日志示例：

```bash
python example.py
```

会看到类似以下的输出（简化版）：

```
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] ================================================================================
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] Initializing LLMEngine
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] Model: ~/huggingface/Qwen3-0.6B/
...
[2026-01-11 17:40:05] [nanovllm.engine.model_runner] [INFO] After KV cache allocation - GPU Memory [Device 0]: Allocated=2.15GB
...
[2026-01-11 17:40:05] [nanovllm.engine.llm_engine] [INFO] Starting generation for 2 prompts
[2026-01-11 17:40:05] [nanovllm.engine.scheduler] [INFO] Scheduled PREFILL batch: size=2, batched_tokens=156
...
[2026-01-11 17:40:08] [nanovllm.engine.llm_engine] [INFO] Generation complete: 42 steps, total_output_tokens=512
```

## 性能调优建议

1. **生产环境**：使用 `WARNING` 级别以减少日志开销
2. **调试/分析**：使用 `DEBUG` 级别查看所有细节
3. **一般使用**：使用 `INFO` 级别获得主要流程信息

## 自定义日志

如果需要自定义日志格式，可以在导入 nanovllm 之前设置：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

from nanovllm import LLM, SamplingParams
```

## 故障排查

使用详细日志可以帮助诊断问题：

- **内存不足**：查看 "GPU Memory" 日志了解内存使用峰值
- **性能问题**：查看 "Elapsed" 日志找出瓶颈
- **调度问题**：查看 Scheduler 的日志了解批次组成
- **缓存效率**：查看 BlockManager 的 cache_hits 了解缓存命中率
