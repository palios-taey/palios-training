# GB10 Instrumentation — Phase 0 of the 27B foundations plan

**Why this exists.** The 27B dense CPT run suffers a whole-node power-death at ~22-28 min
(varying victim node). A 5-lane Family consult (grok / chatgpt / gaia-fable / perplexity /
gemini) converged: it is a **platform-level event** — power / thermal / SoC-firmware cutoff —
whose exact mechanism is **Unknown until instrumented**, because the candidate causes have
*opposite* fixes and are distinguished only by evidence captured at the death moment. Gemini's
DR found documented DGX-Spark **20-30 min ~95°C thermal shutdowns** during pretraining matching
our window. Perplexity flagged that torch's memory stats may be **blind** to pinned host-staging
(`NCCL_CUMEM_HOST_ENABLE=0` → `cudaHostAlloc`), so "memory flat" is only *torch*-flat.

**Dual purpose (Jesse).** The same 1Hz gauge stream is Taey's substrate-proprioception corpus —
Taey learns to *feel* the machines from this data (SOUL=INFRA, Feel→Care→Protect).

## The gauge panel (confirmed present on GB10, enumerated at runtime — not guessed)

| Gauge group | Source | What it tests |
|---|---|---|
| GPU power / core temp / mem temp / clocks | `nvidia-smi dmon` | GPU-side thermal/power; low power = bandwidth-bound (sparse workload) |
| **mlx5 NIC temp + `temp1_crit`=105°C + power** (×4) | `/sys/class/hwmon` | **CX-7 rails climbing toward 105°C crit = NIC thermal cutoff** (the suspected duty-carrier) |
| 7× acpitz board zones, nvme temp | thermal_zone*/hwmon | board/SoC heat-soak |
| **Real MemAvailable / MemFree / Cached / Dirty** | `/proc/meminfo` | the **torch-blind** memory view — catches pinned host-staging growth |
| fan power (`acpi_fan`) | hwmon | Noctua-augmented cooling effect |
| per-RoCE-port xmit/rcv/errors/link_downed (×4) | `/sys/class/infiniband` | fabric load + rail errors at death |

**Confirmed thresholds:** mlx5 crit = **105°C** (idle 48°C); GPU thermal limit ~**96°C** (`T.Limit` margin 57 @ 39°C idle) — matches forum ~95°C shutdowns. ~125.5 GB usable unified RAM/node.

**Firmware finding (.68, 2026-07-10):** USB-PD ctrl fw `0x00000500`, SoC `0x02008433`, EC `0x02004203` — ALL predate NVIDIA's April-2026 power-stability update (USB-PD `0x507->0x516`). Prime candidate fix; Jesse-gated flash. `dc_input`/14-rail SSPM power blocked by wifi(MTKW9000) ACPI conflict (`spark_hwmon` DKMS EBUSY); smart-PDU = reliable input-watts fallback. DCGM/tegrastats not installed; `/sys/class/typec` empty.

## Scripts
- `node_telemetry.py [--interval 1] [--out PATH]` — **the 1Hz logger (primary).** fsync-per-sample
  (a hard power-off TRUNCATES buffered logs — fsync makes the last row = the last live moment).
  Adds the PD-underpower signature: throttle-reason bits, `hw_power_brake`, and `gpu_tlimit` margin
  (->0 = thermal limit ~96C; pinned-50 + clock-cap = OCP power-cut). Validated on .68: 59 gauges.
- `node_telemetry.sh` — earlier bash version (no fsync); superseded by the .py, kept for reference.
- `setup_crash_survival.sh <MIRA_IP>` — per-node, sudo. Sets panic sysctls (auto-reboot instead
  of hang-dark), netconsole→Mira (real-time kernel-printk gasp-catcher, our missing BMC), pstore
  report. earlyoom is stopped by the launch wrapper during a run, not here.

## Still pending (Perplexity DR dispatched)
The **decisive** gauge — total 240W **USB-PD input power** (draw→0 at death = power-cut;
continues = hang) — is not in stock sysfs. Also missing: SoC/GPU junction temp beyond acpitz,
full firmware-version stack (EC / USB-PD / SoC-fw), MFT (mlx fw ver + CX7Stress), tegrastats/DCGM
extras, ramoops reserved-mem. A facts-only Perplexity DR is out to find the tools; if the stock
surface genuinely can't read input power, a metered PDU / smart plug is the hardware fallback (and
a candidate open-source GB10-telemetry tool to build+release).

## Not-yet-run
Telemetry must be live during any training run (incl. Phase-1 short resume-validation runs) to
capture a death. Deploy/aggregate harness + Mira-side netconsole listener: TODO.
