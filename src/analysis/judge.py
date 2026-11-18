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


def _load_keys():
    """Read every GEMINI_API_KEY[_n] from .env, in index order.

    Returns [(name, value)]. Teammates contribute their own keys, so the ring may grow
    between runs; discovery is by pattern rather than a fixed list. Values are never
    returned to anything that logs - only names are ever printed.
    """
    env = Path(".env")
    if not env.exists():
        raise SystemExit(".env not found at the project root")
    found = {}
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(GEMINI_API_KEY(?:_(\d+))?)\s*=\s*(.+?)\s*$", line)
        if m:
            found[m.group(1)] = (int(m.group(2) or 0),
                                 m.group(3).strip().strip('"').strip("'"))
    if not found:
        raise SystemExit("no GEMINI_API_KEY* entries in .env")
    return [(n, v) for n, (_, v) in sorted(found.items(), key=lambda kv: kv[1][0])]


class Budget:
    """Per-key ledger of judge calls against the 500/day cap.

    The API exposes no "calls remaining" endpoint, so counts are kept locally, one entry
    per key name. Self-calibrating rather than clock-based: a daily 429 marks that key
    exhausted, and a later success on it means the quota rolled over. That avoids
    depending on knowing which timezone each key's reset happens in - and with keys from
    different teammates' projects, they need not roll over together.

    Counts are a lower bound: only calls made through this class are seen. A 429 is not
    counted, since rejected requests do not consume quota.
    """

    PATH = Path("results/.judge_budget.json")

    def __init__(self, cap=None):
        self.cap = cap or RPD
        self.state = {"keys": {}}
        if self.PATH.exists():
            try:
                loaded = json.loads(self.PATH.read_text(encoding="utf-8"))
                if "keys" in loaded:
                    self.state = loaded
            except Exception:
                pass

    def _entry(self, name):
        return self.state["keys"].setdefault(
            name, {"count": 0, "exhausted_at": None, "reset_at": None})

    def _save(self):
        self.PATH.parent.mkdir(exist_ok=True)
        self.PATH.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def record(self, name, n=1):
        e = self._entry(name)
        e["count"] += n
        if e["exhausted_at"]:                 # a success means this key rolled over
            e.update({"count": n, "exhausted_at": None,
                      "reset_at": datetime.now().isoformat(timespec="seconds")})
        self._save()

    def mark_exhausted(self, name):
        e = self._entry(name)
        e["count"] = self.cap
        e["exhausted_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()

    def is_exhausted(self, name):
        return bool(self._entry(name)["exhausted_at"])

    def used(self, name):
        return self._entry(name)["count"]

    def remaining(self, name):
        return max(self.cap - self.used(name), 0)

    def total_remaining(self, names):
        return sum(self.remaining(n) for n in names)

    def can_afford(self, names, n, margin=20):
        """Enough headroom ACROSS THE RING to finish a phase, not merely to start one.

        A phase that stalls part-way spends calls on work that has to be redone, so the
        margin is deliberate slack for retries and undercounting.
        """
        return self.total_remaining(names) >= n + margin


class Judge:
    """Rate-limited Gemini client returning parsed JSON.

    Thinking stays ON: types 4 and 5 need the judge to notice something absent - a
    conflict between two passages, or that an answer is not in the corpus - and that is
    exactly the judgement that degrades without it. thinkingBudget is available on this
    model if it ever needs capping.
    """

    def __init__(self, model=MODEL, rpm=RPM, thinking_budget=None):
        self._keys = _load_keys()
        self._redact = [(re.compile(re.escape(v)), n) for n, v in self._keys]
        self.model = model
        self.min_interval = 60.0 / rpm
        self.thinking_budget = thinking_budget
        self.model_version = None
        # Every distinct modelVersion the API has returned. Grades from two versions
        # pooled into one dataset would be a judge confound with no way to detect it
        # afterwards, so this is checked rather than assumed.
        self.model_versions = set()
        self.budget = Budget()
        self.calls = 0
        self.retries = 0
        self.rotations = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self._i = self._first_usable()

    def _first_usable(self):
        """Start on the first key the ledger does not already know is exhausted.

        Rotation is on-exhaustion, not round-robin: a key is used until its daily quota
        is gone, then the ring advances. Round-robin would spread every phase thinly
        across all keys and exhaust them together.
        """
        for i, (name, _) in enumerate(self._keys):
            if not self.budget.is_exhausted(name):
                return i
        return 0

    @property
    def key_name(self):
        return self._keys[self._i][0]

    @property
    def key_names(self):
        return [n for n, _ in self._keys]

    def safe(self, text):
        """Redact EVERY key from anything about to be printed or stored."""
        s = str(text)
        for pat, name in self._redact:
            s = pat.sub(f"<{name}:redacted>", s)
        return s

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
                headers={"x-goog-api-key": self._keys[self._i][1],
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                self.calls += 1
                self.budget.record(self.key_name)
                self.model_version = data.get("modelVersion", self.model)
                self.model_versions.add(self.model_version)
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
                        # Daily cap on THIS key: rotate to the next key that still has
                        # quota and retry the same request. Only when the whole ring is
                        # spent does the run stop.
                        self.budget.mark_exhausted(self.key_name)
                        spent = self.key_name
                        nxt = next((i for i in range(len(self._keys))
                                    if not self.budget.is_exhausted(self._keys[i][0])),
                                   None)
                        if nxt is None:
                            raise RuntimeError(
                                f"ALL {len(self._keys)} KEYS EXHAUSTED for today. "
                                "Work is resumable; rerun after a quota rolls over."
                            ) from None
                        self._i = nxt
                        self.rotations += 1
                        print(f"    [key rotation] {spent} exhausted -> {self.key_name}")
                        continue
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


def reconcile_budget(verbose=True):
    """Probe every key marked exhausted and clear the ones that actually work.

    The ledger self-heals only when a key is USED successfully, but the budget gate
    consults it BEFORE using anything. After a daily rollover every key still reads
    exhausted, so the gate refuses to start a phase the ring can easily afford - which is
    exactly what happened: the ledger said 0 calls remaining while 8 of 9 keys were live.

    One probe per stale key, and it only ever clears flags, never sets them.
    """
    keys = _load_keys()
    b = Budget()
    stale = [(n, v) for n, v in keys if b.is_exhausted(n)]
    if not stale:
        return b
    body = json.dumps({"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}],
                       "generationConfig": {"maxOutputTokens": 16,
                                            "temperature": 0}}).encode()
    cleared = []
    for name, value in stale:
        req = urllib.request.Request(
            ENDPOINT.format(model=MODEL), data=body,
            headers={"x-goog-api-key": value, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                json.load(resp)
            b.record(name)            # a success resets the entry
            cleared.append(name)
        except Exception:
            pass                      # still exhausted; leave the flag alone
    if verbose and cleared:
        print(f"  ledger reconciled: {len(cleared)} key(s) rolled over "
              f"({', '.join(cleared)})")
    return Budget()


def assert_no_key_in(path):
    """Fail loudly if ANY key reached a written artifact."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for name, value in _load_keys():
        assert value not in text, f"{name} LEAKED INTO {path}"
    return True


def assert_single_model_version(rows, field="judge_model_version"):
    """Every graded row must carry the SAME modelVersion.

    Grades produced by two model versions pooled into one dataset would be a judge
    confound: accuracy differences between cells could come from the judge changing
    under the run rather than from the models being compared, and nothing in the
    aggregate would reveal it. Cheap to check, impossible to reconstruct later.
    """
    seen = {r.get(field) for r in rows if r.get(field)}
    assert len(seen) <= 1, (
        f"GRADES SPAN MULTIPLE JUDGE VERSIONS: {sorted(seen)}. "
        "Pooling them is a judge confound - regrade the minority version.")
    return seen.pop() if seen else None


if __name__ == "__main__":
    j = Judge()
    b = j.budget
    print(f"keys in ring  : {len(j.key_names)}  {j.key_names}")
    for n in j.key_names:
        flag = " EXHAUSTED" if b.is_exhausted(n) else ""
        print(f"    {n:<20} used {b.used(n):>4}/{RPD}{flag}")
    print(f"starting key  : {j.key_name}")
    print(f"ring remaining: {b.total_remaining(j.key_names)} calls")
    text, usage = j.ask("Reply with exactly: OK")
    print(f"\nmodel         : {j.model}")
    print(f"modelVersion  : {j.model_version}")
    print(f"reply         : {text.strip()!r}")
    print(f"rate limit    : {RPM} RPM / {RPD} RPD per key")
