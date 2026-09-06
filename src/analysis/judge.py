"""
Stage 8 — Gemini judge client.

The API key lives in .env (gitignored) and is read at use time. It is NEVER hardcoded,
printed, or written to any log, result file or run meta. Every path that surfaces an API
error redacts the key by substring first, because Google's error bodies can echo request
context. Run meta records the model name and the modelVersion the API returns, nothing
else.

Judge: gemini-3.5-flash-lite. Free tier 30 RPM and, MEASURED, 500 requests per day -
GenerateRequestsPerDayPerProjectPerModel-FreeTier = 500, not the 1,500 initially assumed.

gemini-3.8-flash was the first choice on judge strength - it is the stronger model and a
stronger judge raises the ceiling on Cohen's kappa - but its free-tier DAILY quota is
roughly 20 requests. Both cases make the same point: the per-minute limit is the one that
is easy to probe, and the daily limit is the one that actually binds. Budgets here are
sized against the daily figure, and Budget below tracks it.

The cost of that swap is judge quality, and it lands on Cohen's kappa. See
docs/grading_notes.md: kappa is reported prominently with the judge model named beside it,
and if it comes in below 0.6 the README section 7 fallback applies - stricter rubric or
human grading - decided on the measured number rather than assumed away.

External to the grid (Qwen3), satisfying README section 12's no-circularity rule.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL = "gemini-3.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

RPM = 30
# MEASURED, not assumed: GenerateRequestsPerDayPerProjectPerModel-FreeTier = 500.
# The daily cap is what actually binds on this job and it is a third of what was planned
# for, so budgets are sized against this number. It is per-model, so a different model has
# its own 500 - but splitting one grading run across judges would confound the grades, so
# that headroom is deliberately not used.
RPD = 500
MIN_INTERVAL = 60.0 / RPM
MAX_RETRIES = 6


class Budget:
    """Persistent ledger of judge calls against the 500/day cap.

    The API exposes no "calls remaining" endpoint, so the count is kept locally. It is
    SELF-CALIBRATING rather than clock-based: a daily 429 marks the ledger exhausted, and
    a later successful probe means the quota rolled over, so the count resets. That avoids
    depending on knowing which timezone the reset happens in.

    The count is a lower bound on usage - it only sees calls made through this class. A
    429 is not counted, since rejected requests do not consume quota.
    """

    PATH = Path("results/.judge_budget.json")

    def __init__(self, cap=None):
        self.cap = cap or RPD
        self.state = {"count": 0, "exhausted_at": None, "reset_at": None}
        if self.PATH.exists():
            try:
                self.state.update(json.loads(self.PATH.read_text(encoding="utf-8")))
            except Exception:
                pass

    def _save(self):
        self.PATH.parent.mkdir(exist_ok=True)
        self.PATH.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def record(self, n=1):
        self.state["count"] += n
        self._save()

    def mark_exhausted(self):
        self.state["count"] = self.cap
        self.state["exhausted_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()

    def mark_reset(self):
        self.state = {"count": 0, "exhausted_at": None,
                      "reset_at": datetime.now().isoformat(timespec="seconds")}
        self._save()

    @property
    def used(self):
        return self.state["count"]

    @property
    def remaining(self):
        return max(self.cap - self.state["count"], 0)

    def can_afford(self, n, margin=20):
        """Enough headroom to FINISH a phase, not merely to start one.

        A phase that stalls part-way spends calls on work that has to be redone, so the
        margin is deliberate slack for retries and miscounting.
        """
        return self.remaining >= n + margin


def _load_key():
    """Read GEMINI_API_KEY from .env. Never returned to a caller that logs."""
    env = Path(".env")
    if not env.exists():
        raise SystemExit(".env not found at the project root")
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*GEMINI_API_KEY\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not present in .env")


class Judge:
    """Rate-limited Gemini client returning parsed JSON.

    Thinking stays ON: types 4 and 5 need the judge to notice something absent - a
    conflict between two passages, or that an answer is not in the corpus - and that is
    exactly the judgement that degrades without it. thinkingBudget is available on this
    model if it ever needs capping.
    """

    def __init__(self, model=MODEL, rpm=RPM, thinking_budget=None):
        self._key = _load_key()
        self._redact = re.compile(re.escape(self._key))
        self.model = model
        self.min_interval = 60.0 / rpm
        self.thinking_budget = thinking_budget
        self.model_version = None
        self.budget = Budget()
        self.calls = 0
        self.retries = 0
        self._last = 0.0
        self._lock = threading.Lock()

    def safe(self, text):
        """Redact the key from anything about to be printed or stored."""
        return self._redact.sub("<redacted>", str(text))

    def _wait(self):
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()

    def ask(self, prompt, max_output_tokens=8192, temperature=0.0):
        """One call. Returns (text, usage). Raises RuntimeError with a redacted message."""
        cfg = {"maxOutputTokens": max_output_tokens, "temperature": temperature}
        if self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": cfg}).encode()

        for attempt in range(MAX_RETRIES):
            self._wait()
            req = urllib.request.Request(
                ENDPOINT.format(model=self.model), data=body,
                headers={"x-goog-api-key": self._key,
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                self.calls += 1
                self.budget.record()
                if self.budget.state.get("exhausted_at"):
                    self.budget.mark_reset()
                    self.budget.record()
                self.model_version = data.get("modelVersion", self.model)
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts)
                if not text and cand.get("finishReason") == "MAX_TOKENS":
                    raise RuntimeError("judge hit MAX_TOKENS before emitting text "
                                       "- raise max_output_tokens or cap thinking")
                return text, data.get("usageMetadata", {})
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode(errors="replace")
                if exc.code == 429:
                    self.retries += 1
                    if "PerDay" in raw:
                        self.budget.mark_exhausted()
                        raise RuntimeError(
                            "DAILY QUOTA EXHAUSTED (500/day). Work is resumable; "
                            "rerun after the quota rolls over.") from None
                    time.sleep(self._retry_delay(raw, attempt))
                    continue
                if exc.code in (500, 502, 503, 504):
                    self.retries += 1
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {self.safe(raw)[:400]}") from None
            except urllib.error.URLError as exc:
                self.retries += 1
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"exhausted {MAX_RETRIES} retries on {self.model}")

    def _retry_delay(self, raw, attempt):
        """Honour the server's retryDelay when present, else exponential backoff."""
        try:
            for d in json.loads(raw)["error"]["details"]:
                if d.get("@type", "").endswith("RetryInfo"):
                    secs = float(str(d["retryDelay"]).rstrip("s"))
                    return max(secs + 0.5, 1.0)
        except Exception:
            pass
        return min(2 ** attempt + 1, 60)

    def ask_json(self, prompt, **kw):
        """Ask and parse a JSON object or array, tolerating markdown fencing."""
        text, usage = self.ask(prompt, **kw)
        cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(),
                         flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned), usage
        except json.JSONDecodeError:
            m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if not m:
                raise RuntimeError(f"judge did not return JSON: {cleaned[:300]!r}") from None
            return json.loads(m.group(1)), usage


def assert_no_key_in(path):
    """Fail loudly if key material reached a written artifact."""
    key = _load_key()
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    assert key not in text, f"API KEY LEAKED INTO {path}"
    return True


if __name__ == "__main__":
    j = Judge()
    text, usage = j.ask("Reply with exactly: OK")
    print(f"model         : {j.model}")
    print(f"modelVersion  : {j.model_version}")
    print(f"reply         : {text.strip()!r}")
    print(f"thought tokens: {usage.get('thoughtsTokenCount', 0)}")
    print(f"rate limit    : {RPM} RPM / {RPD} RPD ({MIN_INTERVAL:.1f}s between calls)")
