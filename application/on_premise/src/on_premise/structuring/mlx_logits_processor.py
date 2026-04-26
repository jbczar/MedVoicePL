"""Manual MLX logits processor for `lm-format-enforcer`.

`lm-format-enforcer` ships integrations for transformers, llama-cpp,
vLLM, ExLlamaV2, etc. — but as of 0.11.3 there is **no `mlxlm`
submodule**. The library's own `TokenEnforcer` API is backend-agnostic,
so we build the MLX glue here.

Contract (matches mlx-lm's `logits_processors=` kwarg):

    processor(tokens: mx.array, logits: mx.array) -> mx.array

* `tokens` — 1-D array of all tokens fed to the model so far (prompt +
  generated). On step 0 it's the full prompt; on each subsequent step
  one new token is appended.
* `logits` — `(1, vocab)` logits for the next token.
* Returns logits of the same shape, with disallowed tokens masked to
  `-inf`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

if TYPE_CHECKING:
    from lmformatenforcer.characterlevelparser import CharacterLevelParser


def _build_regular_tokens_list(
    tokenizer: Any,
) -> tuple[list[tuple[int, str, bool]], int]:
    """Enumerate the vocab and tag each token with `is_word_start`.

    Mirrors the same pre-processing the transformers integration does:
    decode each id in isolation AND prefixed with token `"0"`. If the
    prefixed decode is longer than the bare decode, the token starts a
    new word (i.e. carries an implicit leading space in SentencePiece).
    """
    if hasattr(tokenizer, "vocab_size"):
        vocab_size = int(tokenizer.vocab_size)
    elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "get_vocab"):
        vocab_size = len(tokenizer._tokenizer.get_vocab())
    elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "vocab_size"):
        vocab_size = int(tokenizer._tokenizer.vocab_size)
    else:
        vocab_size = 32000

    token_0_list = tokenizer.encode("0")
    token_0 = token_0_list[-1] if token_0_list else 0

    regular_tokens: list[tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        decoded_regular = tokenizer.decode([token_idx])
        if not decoded_regular and token_idx != tokenizer.eos_token_id:
            # Skip unprintable / special tokens (except EOS, which we keep).
            continue

        decoded_after_0 = tokenizer.decode([token_0, token_idx])
        decoded_after_0 = (
            decoded_after_0[1:] if decoded_after_0.startswith("0") else decoded_after_0
        )

        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))

    return regular_tokens, vocab_size


def _decode_function(tokenizer: Any, tokens: list[int]) -> str:
    return str(tokenizer.decode(tokens))


def build_mlx_tokenizer_data(tokenizer: Any) -> TokenEnforcerTokenizerData:
    regular_tokens, vocab_size = _build_regular_tokens_list(tokenizer)
    decode_fn = functools.partial(_decode_function, tokenizer)
    eos_id: int | list[int] = tokenizer.eos_token_id
    if hasattr(eos_id, "__iter__") and not isinstance(eos_id, str):
        eos_id = list(eos_id)  # type: ignore[arg-type]

    return TokenEnforcerTokenizerData(
        regular_tokens=regular_tokens,
        decoder=decode_fn,
        eos_token_id=eos_id,
        use_bitmask=False,
        vocab_size=vocab_size,
    )


class MLXLogitsProcessor:
    """Stateful logits-mask wrapping a `TokenEnforcer`.

    `TokenEnforcer.get_allowed_tokens(seq)` returns a `TokenList`
    whose `.allowed_tokens` is a plain Python `list[int]` when
    `use_bitmask=False`. We build a `(vocab,)` mask of `-inf`, set
    allowed indices to `0.0`, and add it to the logits — disallowed
    tokens go to `-inf` and the sampler drops them.
    """

    def __init__(self, token_enforcer: TokenEnforcer) -> None:
        self.token_enforcer = token_enforcer

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        raw = tokens.tolist()
        # Normalize to a flat `list[int]`. mlx-lm 0.31.x passes a 1-D array
        # so `tolist()` already returns `list[int]`, but older versions
        # wrap in `(1, T)` and we'd get `list[list[int]]`.
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            token_sequence: list[int] = raw[0]  # type: ignore[assignment]
        else:
            token_sequence = raw  # type: ignore[assignment]

        allowed = self.token_enforcer.get_allowed_tokens(token_sequence).allowed_tokens
        mask = mx.full(logits.shape[-1:], float("-inf"))
        mask[mx.array(allowed, dtype=mx.int32)] = 0.0

        return logits + mask


def build_mlx_logits_processor(tokenizer: Any, parser: CharacterLevelParser) -> MLXLogitsProcessor:
    tokenizer_data = build_mlx_tokenizer_data(tokenizer)
    token_enforcer = TokenEnforcer(tokenizer_data, parser)
    return MLXLogitsProcessor(token_enforcer)
