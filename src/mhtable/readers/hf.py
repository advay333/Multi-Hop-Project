"""
Hugging Face transformers reader — laptop smoke tests and the fallback when vLLM will not
install (PLAN.md P4). Same prompt, greedy decoding and answer clean-up as the `vllm` reader, but
a different backend, so its `id` (and cache) differ: never mix the two in one comparison.

    reader:
      name: hf
      model: Qwen/Qwen2.5-7B-Instruct
      dtype: float16          # float32 on CPU
      max_new_tokens: 512
      batch_size: 8
      load_in_4bit: false     # bitsandbytes 4-bit, e.g. a 7B reader on the 8 GB laptop GPU

Keys meant for the `vllm` reader are accepted and ignored so one config serves both.
"""

from __future__ import annotations

import time

from mhtable.readers.base import Reader
from mhtable.readers.llm import ReaderStats, chat_prompt, clean_answer, reader_id
from mhtable.registry import register

VLLM_ONLY = {"max_model_len", "tensor_parallel_size", "gpu_memory_utilization", "quantization", "enforce_eager"}


@register("reader", "hf")
class HFReader(Reader):
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", dtype: str = "float16", max_new_tokens: int = 512,
                 batch_size: int = 8, device: str | None = None, load_in_4bit: bool = False,
                 max_input_tokens: int = 6144, seed: int = 0, **ignored):
        unknown = set(ignored) - VLLM_ONLY
        if unknown:
            raise TypeError(f"hf reader: unknown options {sorted(unknown)}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens, self.batch_size, self.max_input_tokens = max_new_tokens, batch_size, max_input_tokens
        self.id = reader_id("hf", model, dtype, max_new_tokens, "bnb4" if load_in_4bit else None)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"  # decoder-only batching
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=kw["dtype"])
            kw["device_map"] = "auto"
        self.lm = AutoModelForCausalLM.from_pretrained(model, **kw)
        if not load_in_4bit:
            self.lm.to(self.device)
        self.lm.eval()
        self.stats = ReaderStats()

    def read_batch(self, questions, subtables, meta=None):
        t0 = time.time()
        prompts = []
        for q, s in zip(questions, subtables):
            text, n, cut = chat_prompt(self.tok, q, s, self.max_input_tokens)
            prompts.append(text)
            self.stats.prompt_tokens += n
            self.stats.prompts_cut += int(cut)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths pad less
        answers: list[str] = [""] * len(prompts)
        eos = self.tok.eos_token_id
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.lm.device)
            with self.torch.no_grad():
                out = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                ids = new[j].tolist()
                self.stats.answers_cut += int(len(ids) >= self.max_new_tokens and eos not in ids)
                answers[i] = clean_answer(self.tok.decode(ids, skip_special_tokens=True))
        self.stats.prompts += len(prompts)
        self.stats.seconds += time.time() - t0
        return answers
