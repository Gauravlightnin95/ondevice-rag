"""
Stage 5 — Verify the frozen retriever still produces identical output.

Re-runnable at any point. Stage 6 must call this before starting the grid, so a run
cannot silently proceed against a drifted retriever — the failure mode this guards is
quiet: a rebuilt index still returns plausible chunk_ids and a grid run against it would
produce numbers that look fine and mean nothing.

Verdicts:
    BYTE_IDENTICAL           artifact hashes and all probe rankings match
    BEHAVIOURALLY_IDENTICAL  a hash moved, every ranking held. Legitimate after a
                             dependency upgrade changes float bits. Proceed, but record
                             what changed alongside the results.
    DRIFTED                  a ranking differs, or structure/parameters differ. Abort.
    ARTIFACTS_MISSING        contract present, binaries absent — the expected state of a
                             fresh clone, since the tag pins the contract not the data.

Hashes and probes catch different things. Hashes catch a rebuilt index. Probes run the
real search() path, so they catch a changed query prefix, RRF constant, candidate depth
or tie-break in retrieve.py — drift that leaves every artifact byte untouched.

    from verify_frozen import verify
    report = verify()
    if report.verdict == "DRIFTED":
        raise SystemExit(report.summary())

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\retrieval\\verify_frozen.py
"""

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "v1"
FROZEN = Path("index") / "frozen" / VERSION
BINARIES = ["faiss.index", "bm25.pkl", "embeddings.npy", "chunk_ids.json", "chunks.jsonl"]
HASH_KEY = {"faiss.index": "faiss_index", "bm25.pkl": "bm25",
            "embeddings.npy": "embeddings", "chunk_ids.json": "chunk_ids",
            "chunks.jsonl": "chunks"}

BYTE_IDENTICAL = "BYTE_IDENTICAL"
BEHAVIOURALLY_IDENTICAL = "BEHAVIOURALLY_IDENTICAL"
DRIFTED = "DRIFTED"
ARTIFACTS_MISSING = "ARTIFACTS_MISSING"


@dataclass
class Report:
    verdict: str
    checks: list = field(default_factory=list)      # (name, ok, detail)
    changed: dict = field(default_factory=dict)     # what moved, for the run log
    failures: list = field(default_factory=list)

    def summary(self):
        lines = [f"frozen retriever {VERSION}: {self.verdict}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if self.changed:
            lines.append("  changed since freeze:")
            for key, (was, now) in self.changed.items():
                lines.append(f"    {key}: {was} -> {now}")
        return "\n".join(lines)

    def as_dict(self):
        """For Stage 6 to write into the run log beside the results."""
        return {"verdict": self.verdict, "freeze_version": VERSION,
                "changed": {k: {"was": w, "now": n} for k, (w, n) in self.changed.items()},
                "failures": self.failures}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(frozen_dir=FROZEN, run_probes=True, verbose=False):
    frozen_dir = Path(frozen_dir)
    fixture_path = frozen_dir / "fixture.json"
    if not fixture_path.exists():
        return Report(DRIFTED, [("fixture", False, f"{fixture_path} not found")],
                      failures=["fixture missing"])
    fx = json.loads(fixture_path.read_text(encoding="utf-8"))

    missing = [n for n in BINARIES if not (frozen_dir / n).exists()]
    if missing:
        return Report(
            ARTIFACTS_MISSING,
            [("artifacts", False, f"absent: {', '.join(missing)}")],
            failures=missing)

    checks, changed, failures = [], {}, []

    # ---- 1. structure and parameters, first so a failure names the cause
    chunk_ids = json.loads((frozen_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    struct_ok = True
    if len(chunk_ids) != fx["corpus"]["n_chunks"]:
        struct_ok = False
        changed["n_chunks"] = (fx["corpus"]["n_chunks"], len(chunk_ids))
        failures.append("chunk count")
    checks.append(("chunk count", struct_ok,
                   f"{len(chunk_ids)} (expected {fx['corpus']['n_chunks']})"))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import retrieve as R
    from build_index import BM25_B, BM25_K1, EMBED_REVISION, QUERY_PREFIX

    params = [
        ("embedding revision", fx["embedding"]["revision"], EMBED_REVISION),
        ("query prefix", fx["query_prefix"], QUERY_PREFIX),
        ("rrf k", fx["fusion"]["k"], R.RRF_K),
        ("candidate depth", fx["fusion"]["candidate_depth"], R.CANDIDATE_DEPTH),
        ("bm25 k1", fx["sparse"]["k1"], BM25_K1),
        ("bm25 b", fx["sparse"]["b"], BM25_B),
    ]
    for name, want, got in params:
        ok = want == got
        checks.append((name, ok, f"{got!r}" + ("" if ok else f" (frozen: {want!r})")))
        if not ok:
            changed[name] = (want, got)
            failures.append(name)

    # ---- 2. artifact hashes
    hashes_ok = True
    for name in BINARIES:
        want = fx["hashes"].get(HASH_KEY[name])
        got = sha256_file(frozen_dir / name)
        ok = want == got
        hashes_ok &= ok
        checks.append((f"hash {name}", ok, got[:16] + ("" if ok else f" (frozen: {want[:16]})")))
        if not ok:
            changed[f"hash:{name}"] = (want[:16], got[:16])

    # environment drift is informational — it explains a hash move, never fails on its own
    env_now = {"python": platform.python_version(),
               "torch": __import__("torch").__version__,
               "faiss": __import__("faiss").__version__,
               "transformers": __import__("transformers").__version__,
               "numpy": __import__("numpy").__version__}
    for key, was in fx["environment"].items():
        if env_now.get(key) != was:
            changed[f"env:{key}"] = (was, env_now.get(key))

    # ---- 3. probes, through the real search path
    probes_ok, n_probe_fail = True, 0
    if run_probes:
        expected = fx["probes"]["expected"]
        k = fx["probes"]["k"]
        for i, probe in enumerate(expected, 1):
            # index_dir is explicit: the probes must run against the directory whose
            # hashes were just checked. Letting search() resolve its own index would
            # verify the hashes of one index and the behaviour of another.
            got = [hit[0] for hit in R.search(probe["question"], k=k,
                                              index_dir=frozen_dir)]
            if got != probe["top_k"]:
                probes_ok = False
                n_probe_fail += 1
                if n_probe_fail <= 3:
                    failures.append(f"probe {probe['qid']}")
                    checks.append((f"probe {probe['qid']}", False,
                                   f"top-{k} differs from the frozen ranking"))
            if verbose and (i % 50 == 0 or i == len(expected)):
                print(f"    probes {i}/{len(expected)}")
        checks.append((f"probes ({len(expected)} queries, top-{k})", probes_ok,
                       "all rankings identical" if probes_ok
                       else f"{n_probe_fail} of {len(expected)} rankings differ"))
    else:
        checks.append(("probes", True, "skipped (run_probes=False)"))

    structural_ok = struct_ok and all(ok for _, ok, _ in checks[:1 + len(params)])
    if not structural_ok or not probes_ok:
        verdict = DRIFTED
    elif hashes_ok:
        verdict = BYTE_IDENTICAL
    else:
        verdict = BEHAVIOURALLY_IDENTICAL

    return Report(verdict, checks, changed, failures)


def main():
    print(f"Verifying frozen retriever {VERSION} in {FROZEN}/\n")
    report = verify(verbose=True)
    print()
    print(report.summary())

    if report.verdict == ARTIFACTS_MISSING:
        print("\nThe freeze contract is present but its artifacts are not. This is the")
        print("expected state of a fresh clone — the tag pins the contract, not the")
        print("corpus-derived data. Rebuild them with:")
        print("    .venv\\Scripts\\python.exe src\\retrieval\\build_index.py")
        print("    .venv\\Scripts\\python.exe src\\retrieval\\freeze.py")
        raise SystemExit(1)

    if report.verdict == BEHAVIOURALLY_IDENTICAL:
        print("\nArtifact bytes moved but every ranking held. Safe to proceed — record the")
        print("'changed' block above alongside any results produced against this index.")
    elif report.verdict == DRIFTED:
        print("\nThe retriever is NOT the one that was frozen. Do not run the grid against")
        print("it: results would not be comparable with anything produced before.")
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
