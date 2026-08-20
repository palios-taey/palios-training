#!/bin/bash
# TOPOLOGY comes from the gitignored fleet.env (see fleet.env.example). NEVER hardcode
# addresses here — the public repo is production infrastructure; topology is deployment config.
[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/fleet.env" ] && . "$(git rev-parse --show-toplevel)/fleet.env"
# 4-node FSDP launcher for Qwen3.6-27B full-parameter CPT. Run on EACH Spark:
#   launch_cpt_qwen36_27b_fsdp.sh <NODE_RANK 0-3> [extra train args]
#
# The communication block below is copied from careers-qwen/launch_4node.sh.
# Do not substitute NCCL, allocator, socket, heartbeat, rank, or master values.

set -uo pipefail
RANK=$1; shift
NCCL_LIB=${SPARK_HOME}/.local/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$NCCL_LIB:${LD_LIBRARY_PATH:-}
# --- validated NCCL block for 4× GB10 dual-rail RoCE CPU-proxy ---
export NCCL_NET_PLUGIN=none          # CRITICAL: AWS-OFI plugin fails on GB10 (proven)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_MERGE_NICS=0          # ChatGPT Mode-B fix: disable dual-port merging (the whole-node crash under sustained collective load survived 12 steps then .80 hard-crashed; MERGE_NICS=0 isolates the dual-rail without falling back to 10GbE). Was 1.
export NCCL_CROSS_NIC=1
export NCCL_BUFFSIZE=8388608
export NCCL_TIMEOUT=1800   # NOTE (GAIA): not a documented NCCL var = silent no-op; real PG timeout is InitProcessGroupKwargs in trainer. Harmless, kept for parity.
# RDMA retry-exhaustion mitigation (IBV_WC_RETRY_EXC_ERR under ~100x reduce-scatter load from
# no_sync-disable; panel Perplexity+Grok converged 2026-07-09). Host-staged CPU-proxy falls behind
# the burst -> QP idles through the retry timers -> node drops off fabric. Extend the retry window +
# spread bursts across QPs. Env-only, reversible.
export NCCL_IB_TIMEOUT=22            # ~4.3s->~17s per retry; ~60-120s total window (was default 20)
export NCCL_IB_RETRY_CNT=7           # hardware-saturating (3-bit QP field)
# NCCL_IB_QPS_PER_CONNECTION left at default 1 (GAIA: neutral-to-negative on 4-node no-ECMP topology;
# the wedge is node power/thermal crash, not fabric routing — real fix is collective-reduction via micro_bsz)
# heartbeat watchdog: without this a hung collective NEVER times out → blocked CUDA stream
# + memory pressure → driver OOM → kernel panic → power-cycle. With it: clean traceback+abort.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120   # 120=clean OOM abort (fleet exp4), not 1800 (30min hang->freeze risk unattended)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # override to INFO to capture topology + tuned bus-bandwidth
# NCCL_DEBUG=INFO is verbose (init topology, rings/trees, per-algo BW estimates) — route it to a
# per-rank file so it doesn't drown the training log. Set NCCL_DEBUG_FILE in the launch env to enable.
[ -n "${NCCL_DEBUG_FILE:-}" ] && export NCCL_DEBUG_FILE
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"   # INIT shows topology detection + tuned bandwidths
# GAIA fragrdma reframe (2026-07-09): the whole-node death (mgmt Realtek + both mlx5 rails die together)
# is NOT a torch-VMM<->NCCL-VMM allocator conflict — it's NCCL silently doing GPUDirect-RDMA (dma-buf MR)
# over torch's expandable-segment pages, which torch later unmaps → NIC DMAs into freed pages → PCIe/kernel
# death. Root: NCCL_NET_GDR_C2C defaults to 1 since 2.27 and OVERRIDES NCCL_NET_GDR_LEVEL on C2C parts (GB10
# = Grace+Blackwell over NVLink-C2C) → GDR silently live despite GDR_LEVEL=0. Fix: force GDR off + disable
# every buffer-registration path so NCCL never registers torch's remappable pages with the NIC.
export NCCL_NET_GDR_C2C=0          # <<< THE one: 2.28 default 1 overrides GDR_LEVEL on C2C parts
export NCCL_NET_GDR_LEVEL=LOC      # (overrides the =0 above; LOC = keep GDR genuinely off)
export NCCL_DMABUF_ENABLE=0        # no dma-buf export of remappable VMM pages
export NCCL_LOCAL_REGISTER=0       # stop ncclCommRegister of allocator segments
export NCCL_GRAPH_REGISTER=0       # stop graph-capture buffer registration
export NCCL_WIN_ENABLE=0           # window registration (cuMem-dependent)
export NCCL_NVLS_ENABLE=0          # no NVSwitch on Spark
export NCCL_CUMEM_HOST_ENABLE=0    # host-staged path: force cudaHostAlloc so host-buf accounting is legible
# NCCL_CUMEM_ENABLE left UNSET (GAIA: auto — =0 pushes NCCL to legacy cudaMalloc which disables safe
# buffer registration and can cause unsafe implicit sync/hang; not the mechanism).
export HF_HUB_DISABLE_XET=1 HF_HOME=${SPARK_HOME}/hf_cache TOKENIZERS_PARALLELISM=false
# TRITON on sm_121 (2026-07-11, Perplexity+Grok consult): Triton's bundled ptxas does NOT know sm_121a →
# kernel compile crashes 'ptxas fatal: Value sm_121a not defined'. Point Triton at the SYSTEM CUDA-13.0
# ptxas (verified: /usr/local/cuda/bin/ptxas supports sm_120,sm_121). THE enabler for ALL Triton on GB10:
# FP8-via-Inductor (torch.compile max_autotune routes _scaled_mm→Triton FP8 when cuBLAS fails), fused-CE,
# DeltaNet kernels. Env-var only, no rebuild. Inert for pure-bf16 runs (no Triton kernels invoked).
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SPARK_HOME}/.triton_cache}"   # pre-cache cubins → no JIT mem spike
# Allocator config is MODEL-RECIPE-SPECIFIC (tutor 2026-07-22, after Jesse: "why are you allowing
# garbage to take up space?"). expandable_segments stays :False in BOTH modes — :True kills this
# RoCE fabric (VMM remapped-page + NCCL-DMA death, f531d64). The DIFFERENCE is the GC threshold:
#   CPT (uniform batch=4): plain :False — the fixed batch shape bounds the per-step peak, cache
#       saturates, no growth. max_split_size_mb:512 BACKFIRED here (OOM step1).
#   LoRA (BATCH_SIZE_PER_RANK=1, DYNAMIC length-sorted padding): allocation size VARIES every step,
#       so the caching allocator hoards a block per size class and NEVER reuses the smaller ones as
#       the length-sorted epoch grows -> reserved climbs monotonically -> the step-490 frag death.
#       garbage_collection_threshold:0.8 makes the allocator AUTO-RELEASE unused cached blocks once
#       reserved exceeds 80% — stopping the garbage at the SOURCE instead of my reactive reclaim.
#       This is the trainer's OWN intended default (train_fsdp_dense_9b.py:20) that the old plain
#       :False override stripped. It is NOT the VMM knob, so it is fabric-safe with :False.
if [ "${LORA_MODE:-0}" = "1" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.8"
    export PYTORCH_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.8"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # CPT: plain — uniform batch bounds the peak
    export PYTORCH_ALLOC_CONF=expandable_segments:False
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"

# MODEL_PATH does NOT default. A DCP continuation constructs the model from the compatible
# 851-tensor training architecture, then RESUME_DELTA restores the learned weights and resumable
# state. The baked 1199-tensor servable and the 851-tensor _hf export use serving parameter names;
# both load without proving that the DCP names landed, so accepting either can silently train the
# wrong weights. The base is therefore an explicit operator decision and its tensor namespace is
# checked before model construction.
export MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH must name the compatible 851-tensor training architecture. A continuation also requires RESUME_DELTA; never use the baked servable or _hf export as MODEL_PATH.}"
if [ -n "${RESUME_DELTA:-}" ] && [ "${LORA_MODE:-0}" != "1" ]; then
    python3 - "$MODEL_PATH/model.safetensors.index.json" <<'PY'
import json
import os
import sys

index_path = sys.argv[1]
if not os.path.isfile(index_path):
    raise SystemExit(f"ERROR: continuation MODEL_PATH has no tensor index: {index_path}")
names = set((json.load(open(index_path)) or {}).get("weight_map") or {})
if (
    len(names) != 851
    or "lm_head.weight" not in names
    or "model.embed_tokens.weight" not in names
    or any(name.startswith("model.language_model.") for name in names)
):
    raise SystemExit(
        "ERROR: continuation MODEL_PATH is not the 851-tensor training namespace; "
        f"observed tensors={len(names)} model.embed_tokens={'model.embed_tokens.weight' in names} "
        f"serving_names={any(name.startswith('model.language_model.') for name in names)}. "
        "Use the training architecture base plus RESUME_DELTA, never a baked artifact."
    )
PY
fi
# LORA_MODE=1 (module training on the frozen CPT base) runs the SFT-pair path instead of CPT:
# SFT_DIR points at the module corpus and the CPT corpus allow-list below is skipped. Unset
# LORA_MODE => byte-identical legacy CPT behavior (sentinel + allow-list enforced).
if [ "${LORA_MODE:-0}" = "1" ]; then
    export SFT_DIR="${SFT_DIR:?ERROR: LORA_MODE=1 requires SFT_DIR (dir of module *.jsonl)}"
    # the SFT path loads ONE file via SFT_JSONL (default $SFT_DIR/tools_sft.jsonl) — point it at the module corpus
    export SFT_JSONL="${SFT_JSONL:-$SFT_DIR/module1_train.jsonl}"
    export CPT_DATA=""
else
    export SFT_DIR="/nonexistent/cpt_mode_sentinel"
    unset SFT_JSONL
    export CPT_DATA="${CPT_DATA:-/var/spark/isma/training/cpt_raw_corpus_train_no_superseded.chunked4096.jsonl}"
fi
export GENERAL_DIR="${GENERAL_DIR:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SPARK_HOME}/training_outputs/cpt_27b_full_ft}"
mkdir -p "$OUTPUT_DIR"

export MAX_SEQ="${MAX_SEQ:-4096}"  # infra-audited default 2026-07-07
export BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-1}"  # infra-audited default 2026-07-07
export TOTAL_STEPS="${TOTAL_STEPS:?ERROR: TOTAL_STEPS must be set for the deployed 27B CPT corpus}"
export SAVE_EVERY="${SAVE_EVERY:-50}"
export SESSION_LIMIT="${SESSION_LIMIT:-250}"
export WARMUP_STEPS="${WARMUP_STEPS:-25}"
export LR="${LR:-1e-5}"
export ADAFACTOR_CLIP_THRESHOLD="${ADAFACTOR_CLIP_THRESHOLD:-1.0}"

# CPT corpus allow-list — enforced ONLY in CPT mode. LORA_MODE=1 trains SFT pairs from
# SFT_DIR, so there is no CPT corpus to validate. (Unset LORA_MODE => unchanged.)
if [ "${LORA_MODE:-0}" != "1" ]; then
case "$CPT_DATA" in
    *cpt_merged_clean.jsonl|*cpt_v3_v2_dense_9b.jsonl|*cpt_v3_v3_dense_9b.jsonl|*cpt_v3_v4_dense_9b.jsonl|*cpt_v3_v4_sorted_dense_9b.jsonl)
        echo "ERROR: CPT_DATA points at a stale or quarantined corpus (contains superseded facts): $CPT_DATA" >&2
        echo "       Use a superseded-EXCLUDED corpus (cpt_raw_corpus_train_no_superseded* or the packed cpt_base_clean_*)." >&2
        exit 1
        ;;
    /var/spark/isma/training/cpt_raw_corpus_train_no_superseded.chunked4096.jsonl)
        ;;
    *comprehensive_v1_packed_*.jsonl)
        # 2026-07-11 THE REAL COMPREHENSIVE BASE (Jesse: KNOW EVERYTHING): repo SOURCE (22 repos 15.4M)
        # + voice (7M) + careers no_superseded (6.8M) + background(x10) + identity + G1-G3. Membrane-clean
        # (draft-dirs + Treasurer superseded-path-list excluded, 0 leak), deduped, coverage-gate PASS.
        ;;
    *cpt_base_clean_packed_*.jsonl)
        # 2026-07-11 (superseded by comprehensive): clean careers+identity+G1-G3 slice — was the throwaway.
        ;;
    *cpt_production_v[0-9]*_packed*.jsonl)
        # CORPUS V1/V2+ production packs (treasurer-registered, sha-gated by pack_production_corpus.py).
        # v2 (161a7ca9): full mandate — 19 repos + career + VOICE + strategy + research, packed sha
        # 9d6e5298, 3686 blocks / 9.44M tok. v1 was the thin 6-slice epoch-1 corpus.
        ;;
    *cpt_production_v1_packed*.jsonl)
        # 2026-07-14 THE PRODUCTION CORPUS (treasurer-registered merge-input set, commits b2e33e3/
        # 745759c/af1f67a): raw_corpus_v4 946r@6e44bd0c + public_repos_v1 890r@948eed2c + careers_kb_v1
        # 316r@ac151e02 + db_worldmodel_v1 33r@bb80a36f + consultations_v1 330r@1a405a83 + recaps_v1
        # 16r@a1079b6f, packed by careers-qwen/pack_production_corpus.py (sha-gated, deterministic):
        # 2531 docs / 1925 blocks / 4.93M tok, output sha f03110dec45e7203... Registered row: treasurer.
        ;;
    *cpt_revenue_jesse*.jsonl)
        # 2026-07-13 REVENUE+JESSE scope-locked (registered: CPT_REVENUE_JESSE_V1_MANIFEST.md, sha c3461005):
        # comprehensive_v1 with constitutional/kernel rows STRIPPED (1817 dropped) → repos-as-capabilities +
        # voice + careers + identity/career-background ONLY. NON-constitutional. 70,215 docs / 11,847 blocks.
        ;;
    *cpt_v6_novoice_packed_2560.jsonl|*cpt_v5_clean_packed_2560.jsonl|*cpt_v3_sanctioned_packed_2560.jsonl)
        # 2026-07-29 cpt_v6_novoice: the 7 authored-document slices, VOICE REMOVED by Jesse
        # directive. 3,033 docs / 2,357 blocks / 6,033,920 tok. Voice was 91.3% of rows and 38.1%
        # of text at 440 chars/row, unlabeled by speaker, 0.77%-filtered, and carried the live
        # credentials. What remains averages 7,149-13,276 chars/doc across hundreds of sourced
        # files. The KB voice guide covers the style need without the transcript dump.
        # 2026-07-29 cpt_v5_clean supersedes cpt_v3_sanctioned: same 8-input basis, repacked
        # after treasurer's union credential scrub moved 5 of the 8 input shas. 3,686 blocks /
        # 9,436,160 tok, output sha c41b803433fafc84..., shrinkage gate 100% vs the previous
        # production corpus. Both names listed because the entry below documents the basis and
        # a rename must not silently drop certification — which it did once tonight, costing a
        # launch: the allowlist named one filename and the pack was written under another.
        # 2026-07-28 THE SANCTIONED RE-EPOCH CORPUS (treasurer sanction given on task-dfa3fd75;
        # packed by careers-qwen/pack_production_corpus.py, every input sha-gated and VERIFIED):
        #   cpt_raw_corpus_v4            946r @ 6e44bd0c02d4ce0c
        #   cpt_careers_kb_v1            316r @ ac151e024a3918fa
        #   cpt_careers_db_worldmodel_v1  33r @ bb80a36f0caf3536
        #   cpt_consultations_v1         330r @ 1a405a832360c770
        #   cpt_recaps_v1                 16r @ a1079b6f37dd848b
        #   voice_cpt_slice_v1_SCRUBBED 31920r @ 0919e05013d3ef69
        # 33,561 docs / 2,511 blocks / 6,428,160 tok, tail cycle-padded (1847 corpus + 713 pad),
        # tail_dropped=0. Output sha256 241f486f5def122ac47e4427a2f4bf8a8d4494e13eb0f264d196eb1aff228592,
        # VERIFIED identical on all 4 Sparks before launch.
        #
        # SUPERSEDED-CLEAN basis: this is exactly the 6 files treasurer classifies `cpt-slice` in
        # build_pairs_manifest.py STATUS — the same STATUS map whose `superseded` class is what this
        # gate exists to exclude. Nothing outside that classification is in the pack.
        #
        # The voice slice is the SCRUBBED derivative and that substitution is load-bearing: the
        # original carried 9 occurrences of 7 distinct live credentials (Anthropic keys, GitHub
        # PATs) captured from transcripts. A credential trained into 27B parameters cannot be
        # removed afterwards — only retrained away. tutor and treasurer scrubbed independently and
        # landed BYTE-IDENTICAL (cmp clean, both sha 0919e05013d3ef69), which is why the value is
        # trusted. Row count preserved 31,920 -> 31,920; 0 credential shapes remain.
        ;;
    *cpt_raw_corpus_train_no_superseded*.jsonl)
        # any packing/chunking of the superseded-excluded careers corpus is allowed
        ;;
    */refresh_v1/MERGED_cpt_refresh_v1_train.jsonl)
        # 2026-07-24 CPT REFRESH v1 MERGE (recipe v0.9): treasurer's scoped refresh slice set
        # (refresh_v1/ASSEMBLY_MANIFEST.json, 12 slices, per-slice 1% holdouts excluded) +
        # phase1_identity_sample 4.04% rider (seed-2560 subsample, 18-block holdout) per Jesse's
        # dilution ruling. 26,030 blocks / 2560, merge sha256 01dab08c011a2f88..., bands:
        # const 3.29 / identity 4.04 / voice 5.06 / replay(prod_v2) 14.0 / dominant ~93.
        # MERGE_MANIFEST.json alongside. Treasurer sanction row REQUESTED 2026-07-24; gate
        # checkpoint QUARANTINED (no bake/serve) until the sanction row lands.
        ;;
    *cpt_refresh_v2_packed.jsonl)
        # 2026-07-25 CPT REFRESH v2 (tutor). Jesse's shape: repetition on what GENERATED
        # training pairs, plus gradual introduction of the rest — no arbitrary repetition.
        # PACKED artifact (pack_corpus.py, tokenizer=prod_v2_ep3_hf, seq_len=2560):
        #   805 blocks x 2560, 2,058,380 tokens from 2461 docs, waste 0.12%, final block
        #   pad_tail=2420. packed sha256 d571ca45261cadee (VERIFIED on all 4 Sparks 2026-07-26).
        #   COVERAGE: 805 blocks / 16 per step = 50.3 steps per epoch.
        #   The RAW {text} form is NOT admissible: PackedCPTDataset reads obj['input_ids']
        #   directly and does no packing, so CPT_PACKED=1 over raw text is a KeyError.
        #
        # ==== TWO CORRECTIONS, 2026-07-27 (tutor, measured; treasurer ruling 1). READ BEFORE ====
        # ==== TRUSTING ANY DESCRIPTION IN THIS BLOCK. Both were FALSE CLAIMS in the record.  ====
        #
        # (1) "Raw source sha256 22f08f3ac1e1192b" — REMOVED above. That digest matches NO FILE
        #     on disk. Measured 2026-07-26: cpt_refresh_v2_clean.jsonl = aff31114a664f07a,
        #     cpt_refresh_v2_train.jsonl = 61524a14c8bab903. Neither is 22f08f3a. The stated
        #     provenance chain raw->packed cannot be verified and must not be cited as if it were.
        #     Root cause: this corpus was assembled AD HOC on 2026-07-25 — no committed builder,
        #     no recorded invocation, and no slice in treasurer/.../training_data/v2/corpus_slices/
        #     (that store holds only the Jul-15 production_v2 lanes). IT IS NOT REPRODUCIBLE.
        #     A corpus we cannot rebuild is one we cannot diff or bisect when a checkpoint regresses.
        #
        # (2) "SUBSTRATE PHYSICS (x3 ...)" — the artifact DOES NOT CONTAIN IT. Decoded all 805
        #     packed blocks with the prod_v2_ep3_hf tokenizer on 2026-07-26: ZERO occurrences of
        #     "SUBSTRATE_PHYSICS", "efficiency identity", or "measured physics of this body";
        #     7 occurrences of "TFLOPS" in 7.7M decoded chars. SUBSTRATE_PHYSICS.md was committed
        #     2026-07-25 (b556a61/2bf8072), AFTER this corpus's content cutoff of 2026-07-24 21:10.
        #     The paragraph below describes INTENDED content that never landed. Retained verbatim
        #     as the intent record, NOT as a description of the artifact.
        #
        # MEASURED COMPOSITION (parsed from the rows, char-weighted — never read off a comment):
        #     sft_delta 64.68% (363 rows) · voice 9.22% (1595) · identity_constitution 9.14% (93)
        #     careers_kb 7.80% (316) · consultations 6.01% (42) · worldmodel 1.76% (33)
        #     recaps 1.39% (16).  57 commits represented, oldest 2026-06-24, NEWEST 2026-07-24 21:10.
        # SANCTION: still NOT obtained. Treasurer ruled 2026-07-27: rebuild the sft_delta lane via
        #     the committed builder (build_cpt_from_sft.py -> pack_production_corpus.py) at cutoff
        #     2026-07-27; the six lanes with no committed builder do not enter a sanctioned corpus
        #     until one exists. Checkpoint from this artifact stays QUARANTINED — no bake, no serve.
        # ==== END CORRECTIONS ====
        #
        # INTENDED (not present — see correction 2): SUBSTRATE PHYSICS x3, added 2026-07-25 per
        #   Jesse ('there needs to be training examples and CPT on this'): the measured constants
        #   of this cluster — GEMM TFLOPS vs clock, collective busbw vs
        #   line rate, the sustained POWER RAMP of a healthy node (26->57W) against a starved one (flat
        #   15.6W), idle-is-identical-at-208MHz, the ~94C wall, the NVTX step split and why NCCL time is
        #   straggler wait rather than bandwidth. Ground truth for a fitted predictive model: feel is a
        #   model evaluated faster than inspection, not a mood, so the body's own physics is CPT content.
        # CORE (x3): the code+docs SFT actually touched — 121 docs from 26 resolved commits
        #   (real diffs) + current-state files, derived by careers-qwen/build_cpt_from_sft.py
        #   across palios-training, treasurer, taeys-hands, the-conductor.
        # GRADUAL: worldmodel 100% / careers_kb 100% / recaps 100% / consultations 15% /
        #   raw_corpus identity+constitution 10% / voice 5%. Deterministic hash-sampled.
        # SUPERSEDED-CLEAN BY CONSTRUCTION: the source list is taken from
        #   build_pairs_manifest.STATUS, admitting only canonical/derived/raw-source. That
        #   mechanically excluded ui_action_rows_v1 + ui_reasoned_rows_v1 (superseded) and
        #   quarantined_structural_preread_rows_v1 + ui_thinking_rows_v1 (QUARANTINED, the
        #   file all 5 lanes said DO NOT TRAIN). A filesystem glob would have swept those in.
        # OPERATOR IDENTITY RETAINED, by the data subject's explicit ruling (Jesse,
        #   2026-07-25): "I do not care about my email addresses or links to conversations.
        #   Those are fine. This is my model and no one can do anything with that." An
        #   earlier pass redacted 30 addresses + 57 conversation URLs; that was reverted.
        #   Redaction is not neutral here — it would train the model to emit a literal
        #   [REDACTED_EMAIL] token exactly where the operator's real contact details belong,
        #   which actively degrades a model expected to complete his applications.
        #   THIS COVERS HIS DATA ONLY. Third-party personal data and live credentials are a
        #   different class and are NOT admitted by this ruling.
        # SANCTION: tutor-assembled, treasurer sanction NOT yet obtained. Same posture as
        #   refresh_v1 above: the run may proceed (it is the production oracle), but the
        #   resulting checkpoint is QUARANTINED — no bake, no serve — until a treasurer
        #   sanction row lands. Standing rule is corpus = treasurer-sanctioned.
        ;;
    *cpt_repos_v1_packed_[0-9]*.jsonl)
        # PUBLIC-REPO SURFACE CORPUS, 2026-08-01. Jesse directive, verbatim: "All the public repos
        # that Taey uses including governance/ need to be trained. Only the public repos they use."
        #
        # WHAT IT IS: cpt_public_repos_v2 ALONE — not the legacy 6-slice blend. The blend drew
        # 65% of its characters from treasurer's PRIVATE consultations tree, which would have
        # trained consultation transcripts rather than the repos and defeated the objective while
        # looking entirely normal.
        #
        # CLEAN SURFACE ONLY: built with the retired-directory exclusion in build_corpus_slices.py,
        # so docs/archive, deprecated, superseded, obsolete and legacy paths are absent. Per Jesse:
        # "The public repos need to be clean. That means nothing out of date ever." Training a
        # current doc alongside its superseded predecessor is the confusion archiving removes.
        # 19 repos including governance; commit-pinned blobs via `git show HEAD:<path>`.
        # TEST FILES ARE EXCLUDED TOO (repo_path_is_test in build_corpus_slices.py): tests/, test/,
        # testing/, __tests__/, spec/ directories and test_*/*_test/conftest files. Per Jesse
        # 2026-08-01: "there are no tests ever... We do production runs only." Training test code
        # would teach a practice the operation does not have. The slice is 1098 rows at
        # sha16 155dc385fb92ed47 AFTER both exclusions; the 1245-row form predates them and is a
        # different pack-set (production_v1) living in a different slices directory.
        #
        # BLOCK LENGTH IS NOT PART OF THIS SANCTION and the pattern deliberately admits any
        # PACK_SEQ: every length is the same 1098 documents and the same 2,731,282 real tokens,
        # differing only in where the block boundary falls. Verified 2026-08-02 — 8192 -> 334
        # blocks, 16384 -> 167, exactly halving, with an identical 4,846-token cycle-pad in the
        # final block either way. What each length DOES change is peak memory, which is why each
        # carries its own full-digest pin below rather than inheriting one.
        #
        # CREDENTIAL-SCANNED: 4 flagged values, every one verified from source as a test fixture
        # or documented placeholder (dev-token, OUTSIDE_MEMORY_SHOULD_NOT_RENDER, change-me).
        #
        # SANCTION: tutor-assembled. Training ownership is tutor's per Jesse 2026-08-01
        # ("Treasurer just does pairs like everyone else... You are in charge").
        ;;
    *cpt_prod_v[0-9]*_packed_[0-9]*.jsonl)
        # PRODUCTION REPO SET, 2026-08-02. Jesse named the repos explicitly after the previous
        # corpus trained education and research repos Taey does not use, verbatim: "those repos are
        # not important... not taey-ed, not research, not local-doge. The main repos that Taey is
        # actually using now."
        #
        # WHAT IT IS: ten repos Taey operates through — palios-training, taeys-hands,
        # claude-code-fleet-orchestrator, apply-machine, claude-code-fleet-notify, dcm, isma-core,
        # taey-presence, governance (CPT-only; its pairs already exist from the 35B full corpus),
        # linkedin. Built by build_corpus_slices.py `public-repos-prod`.
        #
        # IT IS LARGER THAN THE 19-REPO SET IT REPLACES — 1,688 rows against 1,098 — because two
        # surfaces Taey is actually driven through had never been in ANY corpus: apply-machine
        # contributes 709 rows, 42% of this corpus, and linkedin was absent entirely. Training
        # nineteen repos was training less of the right material, not more.
        #
        # SAME EXCLUSIONS as every repo slice: archive/deprecated/superseded/obsolete/legacy path
        # components, and test directories and test_*/*_test files.
        #
        # CREDENTIAL-SCANNED 2026-08-02 with treasurer/scripts/secret_scan.py (the canonical
        # scanner, not an ad-hoc regex set): 0 NAMED matches. 9 entropy candidates, each read and
        # triaged to a code identifier or a public URL — linkedin.com/posts activity ids and a
        # tracxn company id. apply-machine and linkedin had never been scanned before because they
        # had never been in a corpus.
        #
        # SANCTION: tutor-assembled, repo set named by Jesse. Training ownership is tutor's per
        # Jesse 2026-08-01 ("Treasurer just does pairs like everyone else... You are in charge").
        ;;
    *probe_packed_[0-9]*.jsonl)
        # SEQUENCE-LENGTH MEMORY PROBE corpora, 2026-08-01. Admitted DELIBERATELY, not by
        # renaming a file to satisfy the list — the pin block below carries a FULL sha256 for
        # each, which is stricter than most entries above (they pin a name only).
        #
        # WHY THEY EXIST. Nothing in this repository records a full-parameter 27B peak-memory
        # measurement at ANY sequence length, so no packed length is justified by evidence —
        # including the 2560 that was in force. The packer hardcoded SEQ=2560 while this
        # launcher's own audited default is 4096 (line 112), and the packer's constant was the
        # binding value, so the audited window was never exercised. Family consult 2026-08-01:
        # LOGOS declined to choose any length before the curve is measured; HORIZON chose 4096
        # CONDITIONAL on a bounded production run passing first. This is that run.
        #
        # CONTENTS ARE THE PRODUCTION CORPUS, not a synthetic. Same 7 registered slices, same
        # 3,033 documents, same 5,996,544 tokens as the 2560 pack — only the block boundary
        # differs, and tail_dropped=0 at every length. Verified: 4096 -> 1,464 blocks,
        # 8192 -> 732, 16384 -> 366. Halving exactly, as a non-truncating packer must.
        #
        # SCOPE, and this is the binding part: PROBE ONLY. These run with CHECKPOINT_DCP=0 and
        # a session limit in the tens of steps. They produce NO checkpoint, so nothing from them
        # can be baked or served, and no treasurer sanction is implied or claimed. A corpus
        # intended to produce a servable artifact must be sanctioned and pinned on its own row.
        ;;
      *cpt_qwen38_v[0-9]*_nopack_[0-9]*.jsonl)
          # QWEN3.8 BASE-SWAP CORPUS, 2026-08-18. Jesse: "the last CPT round with the production
          # repos updated with the current content."
          #
          # IDENTICAL BLEND to cpt_clean_identity_v1_nopack_8192 — the corpus the model currently in
          # production (s213) actually trained on, per its own live-captured run_config.env
          # (CPT_PATH_FROM_LOG). Same six registered inputs, same document-preserving no-pack builder
          # (cpt_nopack_document_chunks_v2, builder sha ff14a6b4), same MAX_SEQ=8192.
          # FIVE of six inputs are byte-identical to that run's manifest receipts:
          #   cpt_identity_v1 17r@ebdd56e8   cpt_raw_corpus_v4 946r@fd64cb08
          #   cpt_careers_kb_v1 385r@4743ee60   cpt_careers_db_worldmodel_v1 32r@02c203a3
          #   cpt_strategy_research_delta_v1_SCRUBBED 147r@0a81a0af
          # ONLY cpt_public_repos_v2 is refreshed: 1098r@155dc385 -> 1372r@70a1cb24, rebuilt by the
          # production extractor (build_corpus_slices.py public-repos-v2) after fetching every repo to
          # its pinned production ref. 0 NAMED credentials, 0 rows from bundles/, 0 from quarantine.
          #
          # RE-TOKENIZED with the Qwen3.8 tokenizer rather than reusing Qwen3.6-tokenized bytes:
          # vocab.json and merges.txt are byte-identical across the two bases, tokenizer.json and
          # tokenizer_config.json are NOT, and this corpus stores input_ids. Re-tokenizing costs
          # minutes; reusing ids across a changed tokenizer is unverifiable from the artifact after.
          ;;
    *)
        echo "ERROR: CPT_DATA is not an allow-listed superseded-clean corpus: $CPT_DATA" >&2
        echo "       Allowed: cpt_raw_corpus_train_no_superseded*, cpt_base_clean_packed_*." >&2
        exit 1
        ;;
esac

# CONTENT PIN — the allow-list above matches on FILENAME, which certifies nothing about
# what is actually in the file (tutor-codex, 2026-07-25: "the launcher matches filename
# only, so it does not enforce the pinned content identity"). A recorded sha that is only
# a comment is decoration; anything could sit at an allow-listed path. Pin it for real.
# Entries with no pin are unaffected, so this is additive.
case "$CPT_DATA" in
    *cpt_refresh_v2_packed.jsonl) EXPECT_CORPUS_SHA=d571ca45261cadee71a3bf206a0c6b91fc1358881c6a24d767293c198a019735 ;;
    # Sequence-length probe corpora, pinned by FULL digest. Produced by
    # careers-qwen/pack_production_corpus.py at PACK_SEQ=<N> from the 7 registered slices,
    # every input sha-VERIFIED at pack time. Re-packing is deterministic, so a mismatch here
    # means the file is not the artifact these numbers were measured on.
    # public-repo surface corpus — FULL sha256, the strict form
    *cpt_repos_v1_packed_8192.jsonl) EXPECT_CORPUS_SHA=8ae2a486a92c51b288315a921f80cbe253c1ff4242a5d1f3c41db249708022de ;;
    # 16384 pack of the SAME 1098-row slice (155dc385fb92ed47), 2026-08-02. 167 blocks,
    # 2,736,128 tokens — token-for-token identical to the 8192 pack, half the blocks.
    *cpt_repos_v1_packed_16384.jsonl) EXPECT_CORPUS_SHA=b89020fcdc913e3eb5bcd467cc596c7fced11c271d9aac59e15446eba4bf9d31 ;;
    # PRODUCTION repo set at 16384, 2026-08-02. 1,688-row slice (4ee2e63149647997) -> 191 blocks,
    # 3,129,344 tokens, tail_dropped=0.
    # v1 — KILLED MID-RUN AND SUPERSEDED. Pin retained deliberately rather than deleted: 600 of its
    # 1,688 rows were apply-machine/bundles/ (the model's own generated resumes and cover letters,
    # 36% of the corpus) and 5 came from bundles/_QUARANTINE_fabricated/. Killed at step 20/48; no
    # checkpoint was written and nothing was promoted. Keeping the pin means this exact artifact
    # can still be identified if it ever reappears, instead of becoming an unrecognisable file.
    *cpt_prod_v1_packed_16384.jsonl) EXPECT_CORPUS_SHA=0109bcabc90a091e6dd6c38330d1b9a4ed94bdf68d24e3b2e8044261b700985b ;;
    # v2 — bundles/ excluded at the extractor. 1,089-row slice (82a117538c31ecac) -> 150 blocks,
    # 2,457,600 tokens, tail_dropped=0. Verified by SOURCE PATH: 0 rows from bundles/, 0 from
    # quarantine/fabricated. Credential axis clear on two independent grounds (flagged files are
    # under treasurer, and are .jsonl which this corpus does not admit).
    *cpt_prod_v2_packed_16384.jsonl) EXPECT_CORPUS_SHA=04fea4b0ed130ea5b14b1703835dcc0a0171bf2b2ead9c9a72a40be71a6633c9 ;;
    # prod_v3 @ 8192, 2026-08-17. SAME ten-repo production set and SAME extractor as prod_v1/v2 —
    # only the repo content is newer (Jesse: "the same CPT content as current model in production
    # with updated production repo content"). Ten repos fetched to their pinned production refs;
    # taey-presence -> c05d8be, apply-machine -> 13a1b60, governance -> 1f1415f.
    # 1,167 docs / 322 blocks / 2,637,824 tokens, tail_dropped=0. Packer shrinkage gate PASSED at
    # 96% (322 vs the 334 the current production model trained on) and the pack is deterministic —
    # two independent runs emitted this identical digest.
    # THE NAME WAS ALREADY ADMITTED by the cpt_prod_v[0-9]* glob above; without this row the
    # content pin would silently skip, which is the allow-list-as-bypass shape abfd463 hardened.
    *cpt_prod_v3_packed_8192.jsonl) EXPECT_CORPUS_SHA=3ec3587eb155731bfba395c7f270ca6afcc3f2c7fb14bac232984ba60b3ea61e ;;
    *probe_packed_4096.jsonl)  EXPECT_CORPUS_SHA=1dccdd05d9d4776c9f3a2b27909f88c6e18830cf590e1b966c330c458d70ffc1 ;;
    *probe_packed_8192.jsonl)  EXPECT_CORPUS_SHA=d9a7bd45a357c8677c3b29a859ab98f4cdae711c5e988f1b2beb9dc5a3639324 ;;
    *probe_packed_16384.jsonl) EXPECT_CORPUS_SHA=5d3a8e6a84a4a5159177f5ab562d953aeda2a50469af2873aa5df2414f8ba93a ;;
    # 2,922 rows / 5,510,467 tokens / max_seq 8192 / eos 248046. Sidecar manifest carries all six
    # input receipts and its output_sha256 matches these bytes. Verified 4/4 on every rank.
    *cpt_qwen38_v1_nopack_8192.jsonl) EXPECT_CORPUS_SHA=a00ee598a6f6613f1e23e4f2ffaac80ae0b8103c76777cfd237d1978331d489c ;;
    # v2, 2026-08-18 — THE TEN PRODUCTION REPOS (Jesse 2026-08-02, reconfirmed 08-18): apply-machine,
    # claude-code-fleet-notify, claude-code-fleet-orchestrator, dcm, governance, isma-core, linkedin,
    # palios-training, taey-presence, taeys-hands. v1 above used the 19-repo cpt_public_repos_v2 set
    # because the previous CPT round did; that carried taey-ed/local-doge/merge-grade-oss and omitted
    # apply-machine and linkedin. The later directive governs, not the older artifact.
    # 2,717 rows / 5,334,849 tokens. Repo slice 1167r@779b4234 (0 NAMED credentials, 0 bundles rows).
    *cpt_qwen38_v2_nopack_8192.jsonl) EXPECT_CORPUS_SHA=3973c2af608974191c7db2568c008510aa1711bdb714eede31a33fe414576e97 ;;
    *) EXPECT_CORPUS_SHA="" ;;
esac
if [ -n "$EXPECT_CORPUS_SHA" ] && [ -f "$CPT_DATA" ]; then
    # FULL 64-hex digest, not a prefix. A 16-char comparison is 64 bits — fine for a
    # human-readable log line, NOT for a content pin that admits training data
    # (tutor-codex, 6SIGMA recipe fingerprint requires the exact SHA).
    _actual=$(sha256sum "$CPT_DATA" | awk '{print $1}')
    if [ "$_actual" != "$EXPECT_CORPUS_SHA" ]; then
        echo "ERROR: CPT_DATA content pin MISMATCH for $CPT_DATA" >&2
        echo "       expected sha256 ${EXPECT_CORPUS_SHA}" >&2
        echo "       actual   sha256 ${_actual}" >&2
        echo "       The allow-list admits a NAME; this pin admits the CONTENT. Re-pack or" >&2
        echo "       re-pin deliberately — do not rename a file to satisfy the allow-list." >&2
        exit 1
    fi
    echo "  corpus content pin OK (full sha256): ${_actual}"
fi

# SCHEMA GATE — the allow-list admits a NAME, the content pin admits BYTES, and neither
# checks that the loader can actually READ the file. Both shipped DEFAULTS fail this:
# cpt_raw_corpus_train_no_superseded.chunked2560/.chunked4096 are keys=[meta,text] with NO
# input_ids, while PackedCPTDataset does obj['input_ids'] directly and does no packing. So
# CPT_PACKED=1 on either default dies at dataset init, after the reboot, after the deploy,
# after the preflight — the most expensive possible moment to learn it.
# Found by tutor-codex 2026-07-25 reading the actual first row on a node. This is the same
# defect I fixed for one corpus with a content pin and did not check for the others.
if [ "${LORA_MODE:-0}" != "1" ] && [ "${CPT_PACKED:-0}" = "1" ] && [ -f "$CPT_DATA" ]; then
    _schema=$(head -1 "$CPT_DATA" | python3 -c '
import json, sys
try:
    o = json.loads(sys.stdin.readline())
except Exception as e:
    print(f"UNPARSEABLE {e}"); raise SystemExit(0)
print("OK" if isinstance(o.get("input_ids"), list) else "MISSING " + ",".join(sorted(o.keys())))
' 2>/dev/null)
    case "$_schema" in
        OK) echo "  corpus schema OK: first row carries input_ids (packed)" ;;
        MISSING*)
            echo "ERROR: CPT_PACKED=1 but the corpus is NOT packed: $CPT_DATA" >&2
            echo "       first-row keys: ${_schema#MISSING }" >&2
            echo "       PackedCPTDataset reads obj['"'"'input_ids'"'"'] directly and does NO packing," >&2
            echo "       so this dies at dataset init. Pack it first:" >&2
            echo "         python3 dense-9b/data/pack_corpus.py <in> <out> <tokenizer_dir> 2560" >&2
            echo "       Do NOT unset CPT_PACKED to get past this — fixed-shape packing is what" >&2
            echo "       bounds the per-step allocation and prevents the fragmentation OOM." >&2
            exit 1 ;;
        *)  echo "ERROR: could not read the first row of $CPT_DATA ($_schema)" >&2; exit 1 ;;
    esac
fi

if [[ ! -f "$CPT_DATA" ]]; then
    echo "ERROR: CPT_DATA does not exist: $CPT_DATA" >&2
    echo "       Set CPT_DATA to the deployed filtered 27B CPT corpus." >&2
    exit 1
fi
fi

ACCEL_CONFIG="${ACCEL_CONFIG:-$SCRIPT_DIR/../configs/fsdp_dense_27b.yaml}"
if ! ACCEL_EXPORTS="$(
    ACCEL_CONFIG="$ACCEL_CONFIG" python3 - <<'PY'
import os
import shlex
import sys

import yaml

path = os.environ["ACCEL_CONFIG"]
with open(path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

fsdp = config.get("fsdp_config") or {}
version = int(fsdp.get("fsdp_version", 1))

# Common (version-independent) requirements.
common = {
    "distributed_type": "FSDP",
    "mixed_precision": "no",
    "num_machines": 4,
    "num_processes": 4,
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_cpu_ram_efficient_loading": False,
    "fsdp_offload_params": False,
    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    "fsdp_transformer_layer_cls_to_wrap": "Qwen3_5DecoderLayer",
}
if version == 2:
    # FSDP2: reshard_after_forward (bool) replaces sharding_strategy; the v1-only flat-param knobs
    # (backward/forward_prefetch, limit_all_gathers, sync_module_states, use_orig_params) don't apply.
    required = {**common, "fsdp_reshard_after_forward": True}
else:
    required = {
        **common,
        "fsdp_backward_prefetch": "BACKWARD_POST",
        "fsdp_forward_prefetch": False,
        "fsdp_limit_all_gathers": True,
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_sync_module_states": True,
        "fsdp_use_orig_params": True,
    }

actual = {
    "distributed_type": config.get("distributed_type"),
    "mixed_precision": config.get("mixed_precision"),
    "num_machines": config.get("num_machines"),
    "num_processes": config.get("num_processes"),
    **{key: fsdp.get(key) for key in required if key.startswith("fsdp_")},
}
for key, expected in required.items():
    if actual.get(key) != expected:
        print(
            f"ERROR: {path} {key}={actual.get(key)!r}; expected {expected!r}",
            file=sys.stderr,
        )
        sys.exit(1)

exports = {
    "ACCELERATE_USE_FSDP": "true",
    "ACCELERATE_MIXED_PRECISION": "no",
    "FSDP_VERSION": str(version),
    "FSDP_OFFLOAD_PARAMS": str(fsdp["fsdp_offload_params"]).lower(),
    "FSDP_AUTO_WRAP_POLICY": fsdp["fsdp_auto_wrap_policy"],
    "FSDP_TRANSFORMER_CLS_TO_WRAP": fsdp["fsdp_transformer_layer_cls_to_wrap"],
    "FSDP_STATE_DICT_TYPE": fsdp["fsdp_state_dict_type"],
    "FSDP_CPU_RAM_EFFICIENT_LOADING": str(fsdp["fsdp_cpu_ram_efficient_loading"]).lower(),
    "FSDP_ACTIVATION_CHECKPOINTING": str(fsdp.get("fsdp_activation_checkpointing", False)).lower(),
}
if version == 2:
    exports["FSDP_RESHARD_AFTER_FORWARD"] = str(fsdp["fsdp_reshard_after_forward"]).lower()
else:
    exports["FSDP_SHARDING_STRATEGY"] = fsdp["fsdp_sharding_strategy"]
    exports["FSDP_BACKWARD_PREFETCH"] = fsdp["fsdp_backward_prefetch"]
    exports["FSDP_FORWARD_PREFETCH"] = str(fsdp["fsdp_forward_prefetch"]).lower()
    exports["FSDP_USE_ORIG_PARAMS"] = str(fsdp["fsdp_use_orig_params"]).lower()
    exports["FSDP_SYNC_MODULE_STATES"] = str(fsdp["fsdp_sync_module_states"]).lower()
for key, value in exports.items():
    print(f"export {key}={shlex.quote(value)}")
PY
)"; then
    echo "ERROR: failed to load 27B accelerate config: $ACCEL_CONFIG" >&2
    exit 1
fi
eval "$ACCEL_EXPORTS"

NUM_NODES=4
GPUS_PER_NODE=1
MASTER_ADDR=${SPARK_RAIL_MASTER}
MASTER_PORT=29500

echo "FSDP dense 27B CPT on $(hostname) (Rank: $RANK / $((NUM_NODES - 1)))"
echo "  MODEL:  $MODEL_PATH"
echo "  CPT:    $CPT_DATA"
echo "  OUTPUT: $OUTPUT_DIR"
echo "  MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "  ACCEL:  $ACCEL_CONFIG"
echo "  OPTIM:  Adafactor lr=$LR clip=$ADAFACTOR_CLIP_THRESHOLD"
echo "  MEMORY: MAX_SEQ=$MAX_SEQ BATCH_SIZE_PER_RANK=$BATCH_SIZE_PER_RANK"
echo ""

# NSYS_PROFILE_STEP arms cudaProfilerStart/Stop and an NVTX range inside the trainer, but those
# are NO-OPS unless the process runs under a capturing profiler. Wrap only when profiling is
# requested, so an ordinary run launches byte-identically to before.
# --capture-range=cudaProfilerApi makes nsys record exactly the armed optimizer step rather than
# the whole run, which is what keeps the trace small enough to be useful.
# One .nsys-rep per RANK: identifying a straggler is a comparison ACROSS ranks, so the per-rank
# filename is the whole point — a single merged trace cannot say which rank arrived late.
PROFILE_CMD=()
if [ -n "${NSYS_PROFILE_STEP:-}" ] && [ "${NSYS_PROFILE_STEP:-0}" != "0" ]; then
    if ! command -v nsys >/dev/null 2>&1; then
        echo "REFUSE: NSYS_PROFILE_STEP=$NSYS_PROFILE_STEP but nsys is not on PATH." >&2
        echo "        A run that silently skips the capture would report success with no trace." >&2
        exit 1
    fi
    NSYS_OUT="${NSYS_OUT_DIR:-$HOME/cpt27b_logs}/nsys_rank${RANK}_step${NSYS_PROFILE_STEP}"
    mkdir -p "$(dirname "$NSYS_OUT")"
    PROFILE_CMD=(nsys profile
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --trace=cuda,nvtx,osrt
        --sample=none
        --force-overwrite=true
        -o "$NSYS_OUT")
    echo "  NSYS:   capturing rank $RANK step $NSYS_PROFILE_STEP -> ${NSYS_OUT}.nsys-rep"
fi

"${PROFILE_CMD[@]}" python3 -m torch.distributed.run \
    --nnodes="$NUM_NODES" --node_rank=$RANK --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/../trainers/train_fsdp_dense_9b.py" \
    "$@"
