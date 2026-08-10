"""Multi-architecture model loader.

Detects the architecture from `config.json` and returns `(model, tokenizer)`
where the model exposes one uniform interface regardless of what it actually
is::

    forward(input_ids) -> logits
    encode(text) -> list[int]
    decode(ids) -> str
    generate(prompt, max_new_tokens) -> str
    inner_state_dict() -> dict[str, Tensor]     # for the SVD gate
    pad_token_id, stop_token_ids

This is what keeps the scoring code free of tokenizer and architecture
special-casing.

Supported:
  - ChronoGPT (`model_type: chronogpt`, or a legacy config with `model_dim`)
  - Architectures registered under `models/architectures/`
  - Any HuggingFace built-in causal LM

Models are always loaded with `trust_remote_code=False`: no code from a
submitted model directory is ever executed.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from . import architectures  # noqa: F401 — registers custom architectures


class ModelLoadError(RuntimeError):
    """Raised when a submitted model directory cannot be loaded."""


class _HFWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM with the uniform interface."""

    def __init__(self, hf_model, tokenizer):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        # stop tokens: from model config or tokenizer
        stop = getattr(hf_model.config, "eos_token_id", None) or tokenizer.eos_token_id
        if stop is None:
            self.stop_token_ids = set()
        elif isinstance(stop, list):
            self.stop_token_ids = set(stop)
        else:
            self.stop_token_ids = {stop}
        # pad token: for batched scoring, just needs a valid filler token ID
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.bos_token_id = tokenizer.bos_token_id

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.hf_model(input_ids).logits

    def encode(self, text, add_special_tokens=False):
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(ids) if not isinstance(ids, list) else ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def _chat_inputs(self, prompt):
        out = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Complete the sentence with a short answer. Do not repeat the prompt."},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Depending on the transformers version this is a BatchEncoding or a
        # bare tensor.
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return ids.to(self.hf_model.device)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=50, temperature=None, top_k=None, do_sample=None, seed=None):
        """Generate a completion for a single prompt.

        Sampling parameters are forwarded only when explicitly set, so the
        model's own generation config stays in charge by default.
        """
        generate_kwargs = {}
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
        if do_sample is not None:
            generate_kwargs["do_sample"] = do_sample
        if seed is not None:
            # transformers' generate() takes no per-call generator argument,
            # so seeding can only be done through the global RNG.
            torch.manual_seed(seed)

        if getattr(self.tokenizer, "chat_template", None):
            inputs = self._chat_inputs(prompt)
        else:
            inputs = torch.tensor(
                [self.encode(prompt, add_special_tokens=True)], device=self.hf_model.device
            )

        out = self.hf_model.generate(
            inputs, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id, **generate_kwargs
        )
        return self.decode(out[0, inputs.shape[1]:].tolist(), skip_special_tokens=True).strip()

    def parameters(self):
        return self.hf_model.parameters()

    def inner_state_dict(self):
        return self.hf_model.state_dict()


class _ChronoGPTWrapper(nn.Module):
    """Wraps ChronoGPT with the uniform interface."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
        self.stop_token_ids = {self.pad_token_id}

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.model(input_ids)

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=50, top_k=50, seed=42):
        ids = self.encode(prompt)
        device = next(self.model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        for _ in range(max_new_tokens):
            logits = self.model(x)[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, 1, generator=rng)
            next_token = torch.gather(topk_indices, -1, sampled_idx)
            if next_token.item() in self.stop_token_ids:
                break
            x = torch.cat([x, next_token], dim=1)
        return self.decode(x[0, len(ids):].tolist())

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def inner_state_dict(self):
        return self.model.state_dict()


def _is_chronogpt(config: dict) -> bool:
    return config.get("model_type", "").lower() == "chronogpt" or "model_dim" in config


def _wants_half(device) -> bool:
    return getattr(device, "type", str(device)) == "cuda"


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load a model from a local directory. Returns `(model, tokenizer)`."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ModelLoadError(f"No config.json in {model_path}")
    with open(config_path) as f:
        config = json.load(f)

    if _is_chronogpt(config):
        try:
            import tiktoken
        except ImportError as e:
            raise ModelLoadError(
                "ChronoGPT checkpoints need the GPT-2 tokenizer. "
                "Install the extra: pip install 'wigin-tllm[chronogpt]'"
            ) from e
        from .chronogpt import load_model as load_chronogpt

        raw_model = load_chronogpt(model_path, device)
        tokenizer = tiktoken.get_encoding("gpt2")
        return _ChronoGPTWrapper(raw_model, tokenizer), tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Half precision only on GPU. CPU bfloat16 matmuls go through kernels
    # that are chosen by memory alignment: identical weights score slightly
    # differently per load and can sporadically return NaN. float32 on CPU
    # is both correct and reproducible.
    dtype = torch.bfloat16 if _wants_half(device) else torch.float32
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=False
        )
    except ValueError as e:
        if "does not recognize this architecture" not in str(e):
            raise
        # The upstream message suggests upgrading transformers, which is the
        # wrong advice here: custom architectures are only loadable if they
        # have been added to this package.
        raise ModelLoadError(
            f"Unknown architecture {config.get('model_type')!r}. Architectures "
            f"built into transformers work as-is; anything else must be added "
            f"to models/architectures/ (registered here: "
            f"{', '.join(architectures.REGISTERED)})."
        ) from e

    hf_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return _HFWrapper(hf_model, tokenizer), tokenizer
