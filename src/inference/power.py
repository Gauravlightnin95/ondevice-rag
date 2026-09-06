"""
Stage 7 — Power and energy measurement via Windows PDH RAPL counters.

README section 7 originally named HWiNFO. Windows exposes the same RAPL registers
natively as performance counters, and the Energy counter is a MONOTONIC ACCUMULATOR, so
per-query energy is an exact difference between two reads rather than a sampled integral.
That matters here: HWiNFO's free-version CSV logging is ~1s granularity and many queries
are shorter than that (the 0.6B closed-book TTFT is 47 ms). README section 7 has been
updated to name PDH rather than carrying a deviation note.

Rails available:
    RAPL_Package0_PKG    package: cores + GT + uncore + system agent
    RAPL_Package0_PP0    IA (cores)
    RAPL_Package0_PP1    GT (integrated graphics)
    RAPL_Package0_DRAM   memory

There is NO NPU rail, which is why README section 7 specifies a residual at all. PDH also
does not separate system agent, so:

    residual = PKG - PP0 - PP1  =  uncore + system agent + NPU

That is an estimate of an upper bound on NPU power, not a measurement of it, and the
limitation is named rather than footnoted.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\inference\\power.py        # self-test
"""

import subprocess
import time

RAILS = {"pkg": "RAPL_Package0_PKG", "ia": "RAPL_Package0_PP0",
         "gt": "RAPL_Package0_PP1", "dram": "RAPL_Package0_DRAM"}

# The Energy counter's raw unit is NANOJOULES. Three independent lines of evidence, since
# the unit is undocumented and getting it wrong scales every absolute figure:
#
#   1. The sibling Time counter is unambiguously milliseconds - it advanced 8,016 over a
#      measured 8.00 s window.
#   2. At 1e9 the idle package reads 2.43 W and a saturating 8-process scalar load reads
#      8.3 W, both physically plausible for a Lunar Lake 258V.
#   3. The alternative suggested by integrating the PDH Power counter (2.86e8) would put
#      idle at 8.5 W and a math.sqrt loop at 29 W, near the part's TDP. Not credible - and
#      the Power counter is the unreliable reference here, not Energy.
#
# Absolute joules therefore carry this assumption. README section 7 requires figures be
# reported as relative comparisons, which holds under any consistent scale.
UNITS_PER_JOULE = 1e9

_PS = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]


class EnergyReader:
    """Reads the RAPL energy accumulators. Values are raw counter units until scaled.

    A single PowerShell process is kept alive and driven over stdin: spawning one per read
    would cost ~100 ms and load the CPU, which is exactly what we are trying to measure.
    """

    def __init__(self):
        setup = "; ".join(
            f"$c_{k} = New-Object System.Diagnostics.PerformanceCounter("
            f"'Energy Meter','Energy','{v}')" for k, v in RAILS.items())
        setup += "; " + "; ".join(f"$null = $c_{k}.NextValue()" for k in RAILS)
        self.proc = subprocess.Popen(
            _PS + [setup + "; while ($true) { $l = [Console]::ReadLine();"
                   " if ($l -eq 'q') { break };"
                   " $o = @(" + ",".join(f"$c_{k}.NextValue()" for k in RAILS) + ");"
                   " [Console]::WriteLine(($o -join ',')) }"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        self.read()          # prime

    def read(self):
        """Current accumulator value per rail, plus a host timestamp."""
        self.proc.stdin.write("r\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        vals = [float(x) for x in line.split(",")]
        return {"t": time.perf_counter(), **dict(zip(RAILS, vals))}

    def close(self):
        try:
            self.proc.stdin.write("q\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def delta(a, b, units_per_joule=None):
    """Energy between two reads, per rail, plus the residual and elapsed time.

    residual = pkg - ia - gt. DRAM is a separate domain on Intel and is NOT inside PKG,
    so it is reported alongside rather than subtracted.
    """
    scale = units_per_joule or UNITS_PER_JOULE or 1.0
    out = {"seconds": b["t"] - a["t"]}
    for rail in RAILS:
        out[f"{rail}_j"] = (b[rail] - a[rail]) / scale
    out["residual_j"] = out["pkg_j"] - out["ia_j"] - out["gt_j"]
    return out


def power_state():
    """AC or battery. Logged per row: an uncontrolled variable that was not recorded is
    worse than one that was, because it cannot be checked afterwards."""
    try:
        r = subprocess.run(
            _PS + ["$b = Get-CimInstance -Namespace root\\WMI -ClassName BatteryStatus "
                   "-ErrorAction SilentlyContinue; "
                   "if ($b) { \"$($b.PowerOnline),$($b.DischargeRate)\" } else { 'NA,0' }"],
            capture_output=True, text=True, timeout=15)
        online, rate = r.stdout.strip().split(",")
        return {"ac": online.strip().lower() == "true",
                "discharge_mw": float(rate or 0)}
    except Exception as exc:
        return {"ac": None, "discharge_mw": None, "error": str(exc)}


def measure_idle(reader, seconds=10.0):
    """Idle baseline per rail, in raw-units-per-second. Measured, never assumed.

    README section 7 says "baseline idle is around 13 W", which does not match this
    machine (~3.2 W PKG). The discrepancy is recorded rather than reconciled.
    """
    a = reader.read()
    time.sleep(seconds)
    b = reader.read()
    d = delta(a, b, units_per_joule=1.0)
    return {f"{rail}_per_s": d[f"{rail}_j"] / d["seconds"] for rail in RAILS} | {
        "residual_per_s": d["residual_j"] / d["seconds"], "seconds": d["seconds"]}


def check_linearity(reader, durations=(2.0, 4.0, 8.0)):
    """At a fixed load, energy delta must be proportional to window duration.

    An internal consistency check needing no external reference: if the accumulator is a
    true energy counter, implied watts are constant across window lengths. If they drift,
    it is not linear and per-query differences cannot be trusted.
    """
    out = []
    for seconds in durations:
        time.sleep(0.5)
        a = reader.read()
        time.sleep(seconds)
        b = reader.read()
        d = delta(a, b)
        out.append({"seconds": round(d["seconds"], 2),
                    "joules": round(d["pkg_j"], 3),
                    "watts": round(d["pkg_j"] / d["seconds"], 3)})
    w = [o["watts"] for o in out]
    out_spread = (max(w) - min(w)) / min(w) * 100 if min(w) else float("inf")
    return out, out_spread


def _self_test():
    print("PDH RAPL self-test\n")
    print("power state:", power_state())
    with EnergyReader() as r:
        print(f"\nunits per joule: {UNITS_PER_JOULE:,.0f} (nanojoules)")
        print("\nidle baseline (10s), watts:")
        idle = measure_idle(r, 10.0)
        for rail in list(RAILS) + ["residual"]:
            print(f"  {rail:<10} {idle[f'{rail}_per_s'] / UNITS_PER_JOULE:>7.2f} W")

        print("\nlinearity — implied watts must be constant across window lengths:")
        lin, spread = check_linearity(r)
        for o in lin:
            print(f"  {o['seconds']:>4.1f}s  {o['joules']:>7.2f} J  {o['watts']:>6.2f} W")
        print(f"  spread {spread:.1f}%  "
              f"{'OK — accumulator is linear in energy' if spread < 15 else 'NOT LINEAR — do not trust deltas'}")

        print("\ndiscrimination test — idle vs load must separate:")
        a = r.read(); time.sleep(3.0); b = r.read()
        idle_d = delta(a, b)
        a = r.read()
        t0 = time.perf_counter()
        x = 0.0
        while time.perf_counter() - t0 < 3.0:
            x += (x + 2) ** 0.5
        b = r.read()
        busy_d = delta(a, b)
        iw = idle_d["pkg_j"] / idle_d["seconds"]
        bw = busy_d["pkg_j"] / busy_d["seconds"]
        print(f"  idle {iw:6.2f} W    load {bw:6.2f} W    ratio {bw/iw:.2f}x")
        print(f"  {'PASS' if bw/iw > 1.5 else 'FAIL — cannot distinguish idle from load'}")


if __name__ == "__main__":
    _self_test()
