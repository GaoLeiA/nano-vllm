from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.utils.logger import setup_logger

logger = setup_logger(__name__)


class Scheduler:

    def __init__(self, config: Config):
        logger.info(f"Initializing Scheduler: max_num_seqs={config.max_num_seqs}, "
                   f"max_num_batched_tokens={config.max_num_batched_tokens}")
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Track decode steps to reduce logging frequency
        self.decode_step_count = 0
        self.last_batch_size = 0
        logger.info("Scheduler initialized")

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        logger.debug(f"Schedule: waiting={len(self.waiting)}, running={len(self.running)}")
        
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                logger.debug(f"Cannot schedule seq_id={seq.seq_id} for prefill: "
                           f"batched_tokens={num_batched_tokens + len(seq)}/{self.max_num_batched_tokens}, "
                           f"can_allocate={self.block_manager.can_allocate(seq)}")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            seq_info = ', '.join([f"seq_{s.seq_id}(len={len(s)},cached={s.num_cached_tokens})" for s in scheduled_seqs])
            logger.info(f"Scheduled PREFILL batch: size={len(scheduled_seqs)}, "
                       f"batched_tokens={num_batched_tokens}, seqs=[{seq_info}]")
            return scheduled_seqs, True

        # decode
        preempt_count = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                    preempt_count += 1
                else:
                    self.preempt(seq)
                    preempt_count += 1
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        
        # Smart logging: only log at key moments to reduce spam
        self.decode_step_count += 1
        batch_size_changed = len(scheduled_seqs) != self.last_batch_size
        self.last_batch_size = len(scheduled_seqs)
        
        # Log at: first step, every 10 steps, when batch size changes, or if preempted
        should_log_info = (
            self.decode_step_count == 1 or 
            self.decode_step_count % 10 == 0 or 
            batch_size_changed or
            preempt_count > 0
        )
        
        if should_log_info:
            seq_info = ', '.join([f"seq_{s.seq_id}(len={len(s)})" for s in scheduled_seqs])
            logger.info(f"Scheduled DECODE batch (step {self.decode_step_count}): size={len(scheduled_seqs)}, "
                       f"preempted={preempt_count}, seqs=[{seq_info}]")
        else:
            # Still log at DEBUG level for detailed debugging
            seq_info = ', '.join([f"seq_{s.seq_id}(len={len(s)})" for s in scheduled_seqs])
            logger.debug(f"Scheduled DECODE batch (step {self.decode_step_count}): size={len(scheduled_seqs)}, "
                        f"preempted={preempt_count}, seqs=[{seq_info}]")
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        logger.debug(f"Preempting seq_id={seq.seq_id}, len={len(seq)}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        finished_count = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                logger.debug(f"Sequence finished: seq_id={seq.seq_id}, "
                           f"output_len={seq.num_completion_tokens}, "
                           f"reason={'EOS' if token_id == self.eos else 'max_tokens'}")
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_count += 1
        if finished_count > 0:
            logger.debug(f"Postprocess: {finished_count}/{len(seqs)} sequences finished")
