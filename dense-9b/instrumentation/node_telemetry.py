#!/usr/bin/env python3
"""node_telemetry.py — fsync-per-sample 1Hz GB10 gauge logger (supersedes node_telemetry.sh).

WHY fsync (the reason for the Python rewrite): a hard power-off (the dominant GB10 death mode)
TRUNCATES buffered writes — journald/bash-append lose the last rows, which are the ENTIRE point.
Per the GB10 telemetry DR (142 sources): a user-space logger that fsync()s after every write is
"the most reliably effective technique for capturing GB10 death events." So each sample is
flushed + os.fsync'd; the last row on disk is the last live moment before the box goes dark.

WHAT (all confirmed present on GB10; missing sources skipped, never fatal):
  nvidia-smi --query-gpu ->  power, core/mem temp, gr/mem clocks, AND the throttle-reason bits +
     T.Limit margin. The DR's power-cut SIGNATURE: throttle bits all "Not Active" while clocks are
     hard-capped (513-721MHz) + T.Limit pinned to 50C = OCP power-recovery; hw_power_brake = external
     brake. T.Limit -> 0 = thermal limit. These distinguish power-cut vs thermal vs hang at the death.
  sysfs -> acpitz zones, per-NIC mlx5 temp + temp1_crit(105C) + power, nvme/fan power,
     REAL /proc/meminfo (torch-blind view), per-RoCE-port counters.

Dual purpose (Jesse): this stream is Taey's substrate-proprioception corpus (SOUL=INFRA, Feel->Care->Protect).

Usage: node_telemetry.py [--interval 1] [--out PATH]
"""
import argparse, os, subprocess, time, glob, socket, sys

def read(p):
    try:
        with open(p) as f: return f.read().strip()
    except Exception: return ""

def enumerate_sysfs():
    gauges = []  # (col_name, path)
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        tp = f"{z}/temp"
        if os.access(tp, os.R_OK):
            t = read(f"{z}/type")[:10].replace("/", "_") or "z"
            gauges.append((f"tz_{os.path.basename(z).replace('thermal_zone','')}_{t}", tp))
    mlxn = 0
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        nm = read(f"{h}/name")
        if not nm: continue
        tag = nm
        if nm == "mlx5": tag = f"mlx5_{mlxn}"; mlxn += 1
        if os.access(f"{h}/temp1_input", os.R_OK): gauges.append((f"t_{tag}", f"{h}/temp1_input"))
        if os.access(f"{h}/temp1_crit", os.R_OK):  gauges.append((f"tcrit_{tag}", f"{h}/temp1_crit"))
        for pw in (f"{h}/power1_input", f"{h}/power1_average"):
            if os.access(pw, os.R_OK): gauges.append((f"p_{tag}", pw)); break
    for pdir in sorted(glob.glob("/sys/class/infiniband/*/ports/*")):
        cdir = f"{pdir}/counters"
        if not os.path.isdir(cdir): continue
        dev = pdir.replace("/sys/class/infiniband/", "").replace("/ports/", "_p")
        for c in ("port_xmit_data","port_rcv_data","port_xmit_discards","port_rcv_errors","link_downed"):
            if os.access(f"{cdir}/{c}", os.R_OK): gauges.append((f"ib_{dev}_{c}", f"{cdir}/{c}"))
    return gauges

# GPU fields via one nvidia-smi --query-gpu call (nounits). tlimit may be unsupported -> probed.
GPU_BASE = ["power.draw","temperature.gpu","temperature.memory","clocks.gr","clocks.mem",
            "clocks_throttle_reasons.active","clocks_throttle_reasons.hw_slowdown",
            "clocks_throttle_reasons.hw_thermal_slowdown","clocks_throttle_reasons.hw_power_brake_slowdown",
            "clocks_throttle_reasons.sw_power_cap","clocks_throttle_reasons.sw_thermal_slowdown"]

def gpu_query_supported(fields):
    try:
        r = subprocess.run(["nvidia-smi",f"--query-gpu={','.join(fields)}","--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    host = socket.gethostname()
    bootid = read("/proc/sys/kernel/random/boot_id").replace("-","")[:8]
    outdir = os.environ.get("TELEM_DIR", os.path.expanduser("~/telemetry"))
    os.makedirs(outdir, exist_ok=True)
    out = a.out or f"{outdir}/telem_{host}_{bootid}.csv"

    gpu_fields = list(GPU_BASE)
    if gpu_query_supported(GPU_BASE + ["temperature.gpu.tlimit"]):
        gpu_fields.append("temperature.gpu.tlimit")
    gpu_cols = [f.replace("clocks_throttle_reasons.","thr_").replace("temperature.","").replace(".","_")
                for f in gpu_fields]
    sysfs = enumerate_sysfs()
    cols = ["ts_wall","ts_boot"] + gpu_cols + ["mem_avail_kb","mem_free_kb","cached_kb","dirty_kb","load1"] \
           + [c for c,_ in sysfs]

    new = not os.path.exists(out) or os.path.getsize(out) == 0
    f = open(out, "a", buffering=1)
    if new: f.write(",".join(cols) + "\n"); f.flush(); os.fsync(f.fileno())
    sys.stderr.write(f"[telem] {host} {len(cols)} gauges @ {a.interval}s (fsync/sample) -> {out}\n")

    qcmd = ["nvidia-smi", f"--query-gpu={','.join(gpu_fields)}", "--format=csv,noheader,nounits"]
    while True:
        t0 = time.time()
        row = [str(int(t0)), read("/proc/uptime").split(".")[0]]
        try:
            r = subprocess.run(qcmd, capture_output=True, text=True, timeout=4)
            vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")] if r.stdout.strip() else []
        except Exception: vals = []
        row += (vals + [""]*len(gpu_fields))[:len(gpu_fields)]
        mi = {}
        for line in read("/proc/meminfo").splitlines():
            k = line.split(":")[0];
            if k in ("MemAvailable","MemFree","Cached","Dirty"): mi[k] = line.split()[1]
        row += [mi.get("MemAvailable",""), mi.get("MemFree",""), mi.get("Cached",""), mi.get("Dirty","")]
        row.append(read("/proc/loadavg").split(" ")[0])
        for _, p in sysfs: row.append(read(p))
        f.write(",".join(row) + "\n")
        f.flush(); os.fsync(f.fileno())      # <-- survives the hard power-cut
        dt = a.interval - (time.time() - t0)
        if dt > 0: time.sleep(dt)

if __name__ == "__main__":
    main()
