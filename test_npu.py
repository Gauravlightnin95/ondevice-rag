import openvino_genai as ov_genai
import openvino as ov
import time
import re

MODEL_PATH = "models/qwen3-8b-cw"
DEVICES = ["NPU", "GPU", "CPU"]
QUESTION = "What is the capital of France?"
SYSTEM = "Answer concisely and directly. Do not explain your reasoning."
MAX_NEW_TOKENS = 200


def clean(text):
    """Strip thinking tags from Qwen3 output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def thinking_tokens(raw, tok):
    """Count tokens spent inside think blocks."""
    blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if not blocks:
        return 0
    return sum(len(tok.encode(b).input_ids.data[0]) for b in blocks)


def make_pipeline(device):
    if device == "NPU":
        return ov_genai.LLMPipeline(MODEL_PATH, device, NPUW_CACHE_DIR=".npucache")
    return ov_genai.LLMPipeline(MODEL_PATH, device, CACHE_DIR=".ovcache")


def build_prompt(tok):
    """Use /no_think uniformly across all models."""
    try:
        tok.set_chat_template(tok.chat_template)
    except Exception:
        pass

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION + " /no_think"},
    ]
    return tok.apply_chat_template(messages, True)


def run(device, show_prompt=False):
    r = {"device": device, "status": "", "compile_s": None, "gen_s": None,
         "raw_tokens": None, "think_tokens": None, "ms_per_tok": None,
         "raw": "", "answer": ""}

    try:
        t0 = time.time()
        pipe = make_pipeline(device)
        r["compile_s"] = round(time.time() - t0, 1)

        tok = pipe.get_tokenizer()
        prompt = build_prompt(tok)

        if show_prompt:
            print("--- PROMPT ---")
            print(prompt)
            print("--- END PROMPT ---")

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = MAX_NEW_TOKENS
        config.do_sample = False

        t1 = time.time()
        out = pipe.generate(prompt, config)
        elapsed = time.time() - t1

        raw = str(out)
        n_raw = len(tok.encode(raw).input_ids.data[0])

        r["gen_s"] = round(elapsed, 2)
        r["raw_tokens"] = n_raw
        r["think_tokens"] = thinking_tokens(raw, tok)
        r["ms_per_tok"] = round(elapsed * 1000 / max(n_raw, 1), 1)
        r["raw"] = raw
        r["answer"] = clean(raw)
        r["status"] = "OK"

    except Exception as e:
        r["status"] = "FAILED"
        r["answer"] = f"{type(e).__name__}: {e}"

    return r


print(f"OpenVINO version : {ov.get_version()}")
print(f"Devices detected : {ov.Core().available_devices}")
print(f"Model            : {MODEL_PATH}\n")

results = []
for i, device in enumerate(DEVICES):
    print(f"--- {device} ---")
    r = run(device, show_prompt=(i == 0))
    results.append(r)
    if r["status"] == "OK":
        print(f"compile {r['compile_s']}s | generate {r['gen_s']}s "
              f"| raw {r['raw_tokens']} tok | thinking {r['think_tokens']} tok "
              f"| {r['ms_per_tok']} ms/token")
        print(f"RAW    : {r['raw'][:250]}")
        print(f"CLEANED: {r['answer'][:250]}\n")
    else:
        print(f"FAILED: {r['answer']}\n")

print("=" * 84)
print(f"MODEL: {MODEL_PATH}")
print(f"{'Device':<8}{'Status':<9}{'Compile':<10}{'Gen(s)':<9}"
      f"{'RawTok':<9}{'ThinkTok':<10}{'ms/token':<10}")
print("-" * 84)
for r in results:
    def v(x): return x if x is not None else "-"
    print(f"{r['device']:<8}{r['status']:<9}{str(v(r['compile_s'])):<10}"
          f"{str(v(r['gen_s'])):<9}{str(v(r['raw_tokens'])):<9}"
          f"{str(v(r['think_tokens'])):<10}{str(v(r['ms_per_tok'])):<10}")
print("=" * 84)