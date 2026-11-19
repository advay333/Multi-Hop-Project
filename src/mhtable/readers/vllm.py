"""
vLLM reader — the main reader on GPU machines (molab, Kaggle, Prajna; PLAN.md P4, report E.1).

    reader:
      name: vllm
      model: Qwen/Qwen2.5-7B-Instruct
      dtype: float16            # the same precision on every platform (T4 has no bf16)
      max_new_tokens: 512
      max_model_len: 6144       # >= 4,096-token sub-table budget + prompt + answer
      tensor_parallel_size: 1   # 2 on Kaggle's 2x T4
      gpu_memory_utilization: 0.85
      quantization: null        # e.g. awq, with an AWQ checkpoint as `model`

Greedy decoding (temperature 0), so an answer is a pure function of the prompt and the cache in
`CachedReader` is exact. vLLM is imported only when the reader is built (it does not install on
Windows); keys meant for the `hf` reader are accepted and ignored so one config serves both.
"""

from __future__ import annotations

import time

from mhtable.readers.base import Reader
from mhtable.readers.llm import ReaderStats, chat_prompt, clean_answer, reader_id
from mhtable.registry import register

HF_ONLY = {"batch_size", "device", "load_in_4bit", "max_input_tokens"}


@register("reader", "vllm")
class VLLMReader(Reader):
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", dtype: str = "float16", max_new_tokens: int = 512,
                 max_model_len: int = 6144, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85,
                 quantization: str | None = None, enforce_eager: bool = False, seed: int = 0, prompt: str = "p1", **ignored):
        unknown = set(ignored) - HF_ONLY
        if unknown:
            raise TypeError(f"vllm reader: unknown options {sorted(unknown)}")
        import os

        # Decoding is greedy, so FlashInfer's top-k/top-p sampling kernel is never needed. Without
        # its precompiled cubins it is JIT-compiled at start-up, which fails on molab's Blackwell
        # image (system nvcc vs FlashInfer's bundled CCCL headers). Set the variable to 1 to re-enable.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams

        self.model, self.dtype, self.max_new_tokens = model, dtype, max_new_tokens
        self.max_model_len = max_model_len
        self.id = reader_id("vllm", model, dtype, max_new_tokens, quantization, prompt)
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, quantization=quantization,
                       enforce_eager=enforce_eager, seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.prompt = prompt  # prompt version (readers/prompt.py)
        self.stats = ReaderStats()

    def read_batch(self, questions, subtables, meta=None):
        t0 = time.time()
        prompts = []
        for q, s in zip(questions, subtables):
            text, n, cut = chat_prompt(self.tok, q, s, self.max_model_len - self.max_new_tokens, self.prompt)
            prompts.append(text)
            self.stats.prompt_tokens += n
            self.stats.prompts_cut += int(cut)
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)  # outputs keep the input order
        answers = []
        for o in outs:
            c = o.outputs[0]
            self.stats.answers_cut += int(c.finish_reason == "length")
            self.stats.answer_tokens += len(getattr(c, "token_ids", None) or ())
            answers.append(clean_answer(c.text))
        self.stats.prompts += len(prompts)
        self.stats.seconds += time.time() - t0
        return answers
