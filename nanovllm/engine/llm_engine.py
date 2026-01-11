import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.logger import setup_logger, log_gpu_memory

logger = setup_logger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        logger.info("="*80)
        logger.info("Initializing LLMEngine")
        logger.info(f"Model: {model}")
        logger.info(f"Config kwargs: {kwargs}")
        
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        logger.info(f"Configuration: tensor_parallel_size={config.tensor_parallel_size}, "
                   f"max_model_len={config.max_model_len}, "
                   f"max_num_seqs={config.max_num_seqs}, "
                   f"kvcache_block_size={config.kvcache_block_size}, "
                   f"num_kvcache_blocks={config.num_kvcache_blocks}")
        
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        
        # Start worker processes for tensor parallelism
        if config.tensor_parallel_size > 1:
            logger.info(f"Starting {config.tensor_parallel_size - 1} worker processes for tensor parallelism")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            logger.info(f"Started worker process rank={i}, PID={process.pid}")
            self.ps.append(process)
            self.events.append(event)
        
        logger.info("Initializing ModelRunner (rank 0)")
        self.model_runner = ModelRunner(config, 0, self.events)
        
        logger.info(f"Loading tokenizer from {config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        logger.info(f"Tokenizer loaded, EOS token ID: {config.eos}")
        
        logger.info("Initializing Scheduler")
        self.scheduler = Scheduler(config)
        
        log_gpu_memory(logger, "After initialization - ")
        logger.info("LLMEngine initialization complete")
        logger.info("="*80)
        
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
            logger.debug(f"Tokenized prompt: {len(prompt_tokens)} tokens")
        else:
            prompt_tokens = prompt
            logger.debug(f"Using pre-tokenized prompt: {len(prompt_tokens)} tokens")
        seq = Sequence(prompt_tokens, sampling_params)
        self.scheduler.add(seq)
        logger.debug(f"Added request seq_id={seq.seq_id}, prompt_len={len(prompt_tokens)}, max_tokens={sampling_params.max_tokens}")

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        phase = "PREFILL" if is_prefill else "DECODE"
        logger.debug(f"Step [{phase}]: batch_size={len(seqs)}")
        
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if outputs:
            logger.debug(f"Step [{phase}]: {len(outputs)} sequences completed")
        
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        logger.info("-"*80)
        logger.info(f"Starting generation for {len(prompts)} prompts")
        
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Add all requests
        total_prompt_tokens = 0
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                total_prompt_tokens += len(self.tokenizer.encode(prompt))
            else:
                total_prompt_tokens += len(prompt)
            self.add_request(prompt, sp)
        
        logger.info(f"Total prompt tokens: {total_prompt_tokens}")
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        step_count = 0
        total_prefill_time = 0.
        total_decode_time = 0.
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            step_count += 1
            
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / step_time
                    total_prefill_time += step_time
                else:
                    decode_throughput = -num_tokens / step_time
                    total_decode_time += step_time
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        total_output_tokens = sum(len(token_ids) for token_ids in outputs)
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        
        logger.info(f"Generation complete: {step_count} steps, "
                   f"total_output_tokens={total_output_tokens}, "
                   f"prefill_time={total_prefill_time:.2f}s, "
                   f"decode_time={total_decode_time:.2f}s")
        log_gpu_memory(logger, "After generation - ")
        logger.info("-"*80)
        
        return outputs
