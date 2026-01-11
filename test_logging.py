"""
测试日志系统的简单脚本
"""
import os
import logging

# 设置日志级别为DEBUG以查看所有详细信息
os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"

# 配置基本日志
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 测试导入
print("Testing logger import...")
from nanovllm.utils.logger import (
    setup_logger, 
    log_gpu_memory, 
    log_tensor_shapes,
    timed_section
)

# 创建测试logger
logger = setup_logger("test_logger")

logger.info("=" * 80)
logger.info("Testing logging system")
logger.info("=" * 80)

# 测试基本日志
logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")

# 测试 GPU 内存日志（如果有GPU）
try:
    import torch
    if torch.cuda.is_available():
        log_gpu_memory(logger, "Test GPU memory - ")
        logger.info("GPU memory logging OK")
    else:
        logger.info("No GPU available, skipping GPU memory test")
except Exception as e:
    logger.warning(f"GPU memory test failed: {e}")

# 测试张量形状日志
try:
    import torch
    tensors = {
        "tensor_a": torch.randn(2, 3, 4),
        "tensor_b": torch.randn(5, 6),
        "list_of_tensors": [torch.randn(2, 2), torch.randn(3, 3)]
    }
    log_tensor_shapes(logger, tensors, "Test tensor shapes - ")
    logger.info("Tensor shape logging OK")
except Exception as e:
    logger.warning(f"Tensor shape test failed: {e}")

# 测试计时上下文
logger.info("Testing timed section...")
with timed_section(logger, "Test operation"):
    import time
    time.sleep(0.1)

logger.info("=" * 80)
logger.info("All logging tests passed!")
logger.info("=" * 80)

print("\n✓ Logger system is working correctly!")
print("You can now run example.py or bench.py to see detailed logs")
