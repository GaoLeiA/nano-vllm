# 日志优化更新 (2026-01-11)

## 🎯 问题解决

**之前的问题**：DECODE阶段每个step都输出3条INFO日志，导致日志过于冗余

**现在的改进**：采用智能频率控制，大幅减少重复日志

## ✨ 改进内容

### 1. ModelRunner日志优化
- ✅ DECODE阶段的详细信息（张量形状等）从INFO降级到DEBUG
- ✅ 保持PREFILL的INFO日志（较少发生）
- ✅ 需要详细信息时可随时切换到DEBUG级别

### 2. Scheduler智能日志
- ✅ 添加decode step计数器
- ✅ 只在关键时刻输出INFO日志：
  - 第1个decode step
  - 每10个step
  - 批次大小改变时
  - 发生序列抢占时
- ✅ 其他时候使用DEBUG级别

### 3. 文档更新
- ✅ 更新`LOGGING.md`说明新的日志行为
- ✅ 添加INFO和DEBUG级别的对比示例

## 📊 效果对比

### 优化前（每个step 3条INFO日志）
```
[INFO] Scheduled DECODE batch: size=2, preempted=0, seqs=[seq_4(len=133), seq_5(len=139)]
[INFO] [Rank 0] DECODE tensors: input_ids=[2], positions=[2], context_lens=[2], block_tables=[2, 1]
[INFO] [Rank 0] Output: logits=[2, 151936]
[INFO] Scheduled DECODE batch: size=2, preempted=0, seqs=[seq_4(len=134), seq_5(len=140)]
[INFO] [Rank 0] DECODE tensors: input_ids=[2], positions=[2], context_lens=[2], block_tables=[2, 1]
[INFO] [Rank 0] Output: logits=[2, 151936]
[INFO] Scheduled DECODE batch: size=2, preempted=0, seqs=[seq_4(len=135), seq_5(len=141)]
... (每个step都重复) ❌
```

### 优化后（周期性INFO日志）
```
[INFO] Scheduled DECODE batch (step 1): size=2, preempted=0, seqs=[seq_4(len=79), seq_5(len=79)]
... (步骤2-9：静默或仅DEBUG)
[INFO] Scheduled DECODE batch (step 10): size=2, preempted=0, seqs=[seq_4(len=88), seq_5(len=88)]
... (步骤11-19：静默或仅DEBUG)
[INFO] Scheduled DECODE batch (step 20): size=2, preempted=0, seqs=[seq_4(len=98), seq_5(len=98)]
[INFO] Sequence finished: seq_id=4, output_len=100
[INFO] Sequence finished: seq_id=5, output_len=106
[INFO] Generation complete: 52 steps, total_output_tokens=206
... (简洁清晰) ✅
```

**日志减少比例**：约 **90%** 的INFO日志减少（从每step 3条 → 每10步1条）

## 🔧 使用方法

### 默认（INFO级别，简洁）
```bash
python example.py
```

### 详细调试（DEBUG级别，看所有细节）
```bash
# Windows
$env:NANOVLLM_LOG_LEVEL="DEBUG"
python example.py

# Linux/Mac
export NANOVLLM_LOG_LEVEL=DEBUG
python example.py
```

## 📝 修改的文件

1. `nanovllm/engine/model_runner.py` - DECODE日志降级到DEBUG
2. `nanovllm/engine/scheduler.py` - 添加智能频率控制
3. `LOGGING.md` - 更新文档说明

## ⚡ 立即生效

无需额外配置，只需正常运行：
```bash
python example.py  # INFO级别，简洁日志
python bench.py    # INFO级别，适合benchmark
```

如需查看所有细节：
```bash
NANOVLLM_LOG_LEVEL=DEBUG python example.py
```

---

**优化完成！** 现在日志更加简洁易读，同时保留了完整的调试能力。 ✨
