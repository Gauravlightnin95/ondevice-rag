"""
Stage 6 — OpenVINO GenAI pipeline setup, parameterised by model path.

The device handling, cache properties, chat-template quirks and /no_think prompting are
test_npu.py's, which established them against the hardware (README section 3). They are
reused here rather than rewritten; test_npu.py itself is left untouched as the record of
that characterisation.

Two helpers differ from test_npu.py's, deliberately. README section 2 specifies THREE
regexes in clean(); test_npu.py has two, omitting the unclosed-tag case. README section 3
records that the 0.6B "opens a <think> tag and never closes it", so with the two-regex
version that model's entire output would survive as if it were an answer, and
thinking_tokens() would report 0 for output that is wholly thinking. README section 2 is
explicit that hiding thinking cost is not acceptable, so both handle the unclosed case.

Run from C:\\ondevice-rag
"""

import re

import openvino_genai as ov_genai

# README section 2: uniform across all four models. Varying it by size would confound
# the size axis, which is the experiment's main effect.
SYSTEM = "Answer concisely and directly. Do not explain your reasoning."
NO_THINK = " /no_think"


def make_pipeline(model_path, device, max_prompt_len=None):
    """Cache property differs by device. Passing NPUW_CACHE_DIR to GPU or CPU raises.

    MAX_PROMPT_LEN is NPU-only and mandatory for RAG. The NPU pipeline statically
    allocates for a declared maximum prompt; at OpenVINO GenAI's default of 1024 tokens it
    CANNOT run k=5 RAG on this corpus at all — prompts are 2,889-4,192 tokens and every
    one raises. Declaring a larger buffer is not free: it costs ~13.6% latency on every
    short prompt and 22x compile time, a tax paid even by queries carrying no retrieved
    context. See docs/inference_notes.md — this is an H1 finding, not a config note.
    """
    if device == "NPU":
        kwargs = {"NPUW_CACHE_DIR": ".npucache"}
        if max_prompt_len:
            kwargs["MAX_PROMPT_LEN"] = max_prompt_len
        return ov_genai.LLMPipeline(model_path, device, **kwargs)
    return ov_genai.LLMPipeline(model_path, device, CACHE_DIR=".ovcache")


def make_config(max_new_tokens):
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False                 # greedy, README section 8
    # The prompt is built manually via the tokenizer below. Leaving this on would apply
    # the chat template a second time and silently change every prompt.
    config.apply_chat_template = False
    return config


def build_prompt(tok, user_content, no_think=True):
    """Apply the chat template. add_generation_prompt is POSITIONAL, not a keyword.

    no_think=False is used ONLY by the 0.6B diagnostic, which sits outside the main grid.
    Every row in the grid itself uses no_think=True, uniformly across all four models.
    """
    try:
        tok.set_chat_template(tok.chat_template)
    except Exception:
        pass
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content + (NO_THINK if no_think else "")},
    ]
    return tok.apply_chat_template(messages, True)


CLOSED_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# <think> followed by a BLANK LINE and never closed. This is /no_think working: Qwen3
# emits an empty thinking block "<think>\n\n</think>" and the closer is missing from the
# decoded text. What follows is the ANSWER.
ELIDED_EMPTY = re.compile(r"<think>[ \t]*\r?\n[ \t]*\r?\n")
UNCLOSED_TEXT = re.compile(r"<think>.*$", re.DOTALL)


def clean(text):
    """Strip thinking from Qwen3 output.

    README section 2 gives a three-regex version whose second rule deletes everything
    after an unclosed <think>. Measured against 138 real outputs that rule is wrong for
    the 8B and destroys 31 of its answers, so this deviates deliberately. The three cases
    are genuinely distinct and the signature separates them with zero overlap:

      <think>...</think>   closed block          strip it            (1.7B, 90/138 rows)
      <think>\\n\\n + text   EMPTY block, closer   keep what follows   (8B, 31 rows,
                           dropped by decoder     -- it is the answer  finish=STOP)
      <think>\\n + text     genuine reasoning,    strip all of it     (0.6B, 25 rows,
                           never closed                                finish=LENGTH)

    The rule keys on the /no_think token pattern, not on model identity — cleaning that
    varied by model size would confound the size axis.
    """
    text = CLOSED_BLOCK.sub("", text)
    if "<think>" in text:
        if ELIDED_EMPTY.search(text):
            text = ELIDED_EMPTY.sub("", text)      # empty block: the answer follows
        else:
            text = UNCLOSED_TEXT.sub("", text)     # real unclosed reasoning: drop it
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def thinking_tokens(raw, tok):
    """Tokens spent inside think blocks, mirroring clean()'s three cases.

    An elided EMPTY block contributes zero, because nothing was thought. A genuine
    unclosed block contributes all of it — test_npu.py's version matches only
    <think>...</think> and would report 0 for output that is entirely thinking, which
    README section 2 is explicit about not hiding.
    """
    blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    rest = CLOSED_BLOCK.sub("", raw)
    if "<think>" in rest and not ELIDED_EMPTY.search(rest):
        tail = re.search(r"<think>(.*)$", rest, flags=re.DOTALL)
        if tail:
            blocks.append(tail.group(1))
    return sum(count_tokens(b, tok) for b in blocks)


def count_tokens(text, tok):
    if not text:
        return 0
    return len(tok.encode(text).input_ids.data[0])
