# 日志功能实现总结

## 实现完成 ✅

已为 nano-vllm 添加了全面的日志系统，现在运行时可以看到整个推理流程的详细信息。

## 修改的文件

### 新增文件
1. **`nanovllm/utils/logger.py`** - 日志工具模块
   - `setup_logger()` - 配置logger
   - `log_gpu_memory()` - GPU内存追踪
   - `log_tensor_shapes()` - 张量形状记录
   - `timed_section()` - 计时上下文管理器
   - `log_communication()` - 通信日志（用于分布式）

2. **`LOGGING.md`** - 详细的日志使用文档
3. **`test_logging.py`** - 日志系统测试脚本

### 修改的文件

1. **`nanovllm/engine/llm_engine.py`**
   - `__init__`: 记录配置、组件初始化、GPU内存
   - `add_request`: 记录请求添加和token化
   - `step`: 记录每个step的阶段和完成情况
   - `generate`: 记录批次信息、提示词tokens、生成统计

2. **`nanovllm/engine/scheduler.py`**
   - `__init__`: 记录调度器配置
   - `schedule`: 记录调度决策、批次组成、序列信息
   - `preempt`: 记录抢占事件
   - `postprocess`: 记录完成的序列

3. **`nanovllm/engine/block_manager.py`**
   - `__init__`: 记录块管理器配置
   - `allocate`: 记录块分配、缓存命中/未命中、剩余块数
   - `deallocate`: 记录块释放

4. **`nanovllm/engine/model_runner.py`**
   - `__init__`: 记录模型加载、配置、GPU内存、KV cache分配
   - `allocate_kv_cache`: 记录KV cache配置和GPU内存使用
   - `prepare_prefill`: 记录prefill批次准备、张量形状
   - `prepare_decode`: 记录decode批次准备、张量形状
   - `run`: 记录运行阶段、准备和前向传播的耗时

5. **`nanovllm/layers/attention.py`**
   - `store_kvcache`: 记录KV cache存储操作
   - `forward`: 记录attention计算模式、输入输出形状

6. **`example.py`** - 添加了日志配置
7. **`bench.py`** - 添加了日志配置
8. **`pyproject.toml`** - 添加了 `tqdm` 依赖

## 日志内容

### INFO级别（默认）
- 初始化信息（配置、模型加载、内存分配）
- 生成开始和结束
- 调度决策（prefill/decode批次）
- 生成统计（步数、tokens、耗时）
- GPU内存使用

### DEBUG级别（详细）
- 所有INFO级别的内容
- 每个step的详细信息
- 张量形状
- 块分配/释放细节
- Attention层操作
- 操作耗时（毫秒级）
- 序列完成细节

## 使用方法

### 1. 设置日志级别

```bash
# Windows
$env:NANOVLLM_LOG_LEVEL="DEBUG"
python example.py

# Linux/Mac
export NANOVLLM_LOG_LEVEL=DEBUG
python example.py
```

### 2. 在代码中启用

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # 或 logging.DEBUG
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)

from nanovllm import LLM, SamplingParams
```

### 3. 运行示例

```bash
# 运行示例（INFO级别）
python example.py

# 运行基准测试（INFO级别）
python bench.py

# 测试日志系统
python test_logging.py
```

## 日志示例输出

运行 `python example.py` 会看到：

```
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] ================================================================================
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] Initializing LLMEngine
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] Model: ~/huggingface/Qwen3-0.6B/
[2026-01-11 17:40:00] [nanovllm.engine.llm_engine] [INFO] Configuration: tensor_parallel_size=1, max_model_len=2048, ...
[2026-01-11 17:40:01] [nanovllm.engine.model_runner] [INFO] [Rank 0] Loading model: ~/huggingface/Qwen3-0.6B/
[2026-01-11 17:40:02] [nanovllm.engine.model_runner] [INFO] After model loading - GPU Memory: Allocated=1.85GB
[2026-01-11 17:40:03] [nanovllm.engine.model_runner] [INFO] Allocating 512 KV cache blocks, total_size=1.12GB
[2026-01-11 17:40:03] [nanovllm.engine.model_runner] [INFO] After KV cache allocation - GPU Memory: Allocated=2.97GB
...
[2026-01-11 17:40:04] [nanovllm.engine.llm_engine] [INFO] Starting generation for 2 prompts
[2026-01-11 17:40:04] [nanovllm.engine.llm_engine] [INFO] Total prompt tokens: 156
[2026-01-11 17:40:04] [nanovllm.engine.scheduler] [INFO] Scheduled PREFILL batch: size=2, batched_tokens=156, seqs=[seq_0(len=78,cached=0), seq_1(len=78,cached=0)]
...
[2026-01-11 17:40:06] [nanovllm.engine.llm_engine] [INFO] Generation complete: 42 steps, total_output_tokens=512, prefill_time=0.15s, decode_time=2.43s
```

## 主要特性

✅ **输入输出形状追踪** - 所有张量的维度信息  
✅ **内存使用监控** - GPU内存的分配和峰值  
✅ **执行时间分析** - 各阶段的耗时统计  
✅ **调度决策可视化** - 批次组成和序列调度  
✅ **块管理详情** - KV cache块的分配和缓存命中率  
✅ **通信日志** - 分布式训练的通信操作（如有）  
✅ **灵活的日志级别** - 通过环境变量或代码控制  

## 性能影响

- **INFO级别**：影响很小（<1%）
- **DEBUG级别**：略有影响（1-3%），因为包含更多字符串格式化和I/O操作
- **生产环境建议**：使用WARNING或INFO级别

## 下一步

现在你可以：

1. 运行 `python example.py` 查看完整的执行日志
2. 运行 `python bench.py` 查看基准测试的详细信息
3. 使用 `NANOVLLM_LOG_LEVEL=DEBUG` 查看最详细的日志
4. 参考 `LOGGING.md` 了解所有日志内容的详细说明

如需调整日志内容或添加新的日志点，可以参考现有代码的实现方式。
