"""Shared vLLM loading + generation helpers for the RAG refutation pipeline."""
import re
from typing import Callable, Optional

from huggingface_hub import snapshot_download


def resolve_model_path(model_name: str) -> str:
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name.split("/")[-1])


def load_llm(model_name: str, seed: int = 42, max_model_len: int = 6144, gpu_mem: float = 0.6,
             max_num_seqs: int | None = None):
    from vllm import LLM
    model_path = resolve_model_path(model_name)
    print(f"Loading {model_name} <- {model_path}")
    extra = {} if max_num_seqs is None else {"max_num_seqs": max_num_seqs}
    llm = LLM(
        model=model_path,
        dtype="float16",
        seed=seed,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        **extra,
    )
    return llm


def build_chat_prompts(tokenizer, messages_list: list[list[dict]]) -> list[str]:
    prompts = []
    for msgs in messages_list:
        try:
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            p = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs) + "<|assistant|>\n"
        prompts.append(p)
    return prompts


def generate_with_retry(
    llm,
    tokenizer,
    messages_list: list[list[dict]],
    parse_fn: Callable[[str], tuple],
    seed: int,
    max_tokens: int = 512,
    max_retries: int = 2,
) -> list[dict]:
    """
    parse_fn(text) -> tuple whose first element is None iff the parse failed
    (used to trigger a retry at higher temperature). Returns list of
    {"raw": str, "parsed": tuple, "parse_success": bool} aligned with messages_list.
    """
    from vllm import SamplingParams

    prompts = build_chat_prompts(tokenizer, messages_list)
    n = len(prompts)
    results: list[Optional[dict]] = [None] * n
    retry_indices = list(range(n))

    for attempt in range(max_retries + 1):
        if not retry_indices:
            break
        batch_prompts = [prompts[i] for i in retry_indices]
        sp = SamplingParams(
            temperature=min(0.3 * attempt, 0.6),
            max_tokens=max_tokens,
            seed=seed + attempt,
        )
        outputs = llm.generate(batch_prompts, sp)
        still_bad = []
        for idx, out in zip(retry_indices, outputs):
            text = out.outputs[0].text.strip()
            parsed = parse_fn(text)
            ok = parsed[0] is not None
            if ok or attempt == max_retries:
                results[idx] = {"raw": text, "parsed": parsed, "parse_success": ok}
            else:
                still_bad.append(idx)
        retry_indices = still_bad
        if still_bad:
            print(f"    retry {attempt + 1}/{max_retries}: {len(still_bad)} parse failures")

    return results
