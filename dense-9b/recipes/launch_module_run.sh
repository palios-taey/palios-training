#!/bin/bash
# launch_module_run.sh — deploy a training runtime from IMMUTABLE COMMIT OBJECTS.
#
# WHY (tutor-codex final review, 2026-07-25): a HIGH Git-provenance race exists when a
# launcher checks a working tree once and then deploys FROM that mutable tree. The bytes
# verified and the bytes shipped are two different reads of a file that can change in
# between — and on a shared remote, two nodes can receive different content from one
# "verified" source.
#
# My own launcher was worse than the case they found. It lived in a session scratch
# directory (untracked, unversioned, unshareable), deployed from mutable scratch copies,
# and hardcoded three node IPs. So the thing that decides what runs on the cluster was
# itself the least governed artifact in the pipeline.
#
# THE FIX, and it is structural rather than a tighter check:
#   deploy from `git show <SHA>:<path>` — a commit object is immutable by construction.
#   The same SHA yields the same bytes on every node, on every rerun, forever. There is no
#   window between verify and ship because the object cannot change.
#
# TOPOLOGY-FREE: node addresses come from the environment, never from this file, so it is
# safe to track and survives a flip to public.
#
# Usage:
#   SPARK_NODES="h1 h2 h3 h4" MASTER_ADDR=<rank0-fabric-addr> \
#   DEPLOY_SHA=<commit> CORPUS=<path> CORPUS_SHA256=<full 64-hex> \
#   bash launch_module_run.sh
set -uo pipefail

: "${SPARK_NODES:?ERROR: SPARK_NODES must list the nodes in rank order}"
: "${MASTER_ADDR:?ERROR: MASTER_ADDR must identify the rank-0 training-fabric address}"
: "${DEPLOY_SHA:?ERROR: DEPLOY_SHA must pin the commit the runtime is deployed FROM}"
: "${CORPUS:?ERROR: CORPUS must be an absolute path to the packed corpus}"
: "${CORPUS_SHA256:?ERROR: CORPUS_SHA256 must be the full 64-hex digest of CORPUS}"

# CUMULATIVE ADAPTER (optional; if set, BOTH must be set). tutor-codex 2026-07-25:
# REQUIRE_LORA_INIT_PARITY proves the four ranks AGREE WITH EACH OTHER — it does NOT bind
# a known-expected adapter. Four ranks holding the same WRONG adapter pass it cleanly.
# That is a CONSISTENCY check masquerading as a CORRECTNESS check, and it is invisible
# precisely when the mistake is uniform. A pinned expected digest converts "they match"
# into "they match the thing we intended".
ADAPTER="${ADAPTER:-}"
ADAPTER_SHA256="${ADAPTER_SHA256:-}"
if [ -n "$ADAPTER" ] || [ -n "$ADAPTER_SHA256" ]; then
    [ -n "$ADAPTER" ] && [ -n "$ADAPTER_SHA256" ] || {
        echo "ERROR: ADAPTER and ADAPTER_SHA256 must be set together — an adapter without a" >&2
        echo "       pinned digest is a path, not a provenance claim." >&2
        exit 1; }
fi

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
REMOTE_ROOT="${REMOTE_ROOT:?ERROR: REMOTE_ROOT must be the deploy root on each node}"
REMOTE_USER="${REMOTE_USER:?ERROR: REMOTE_USER must be the node account}"
read -r -a NODES <<< "$SPARK_NODES"
[ "${#NODES[@]}" -eq 4 ] || { echo "ERROR: SPARK_NODES must name exactly four nodes" >&2; exit 1; }

say(){ echo "[deploy $(date -u +%H:%M:%S)] $*"; }

# The two runtime files, resolved as COMMIT OBJECTS. Never read from the working tree.
TRAINER_PATH=dense-9b/trainers/train_fsdp_dense_9b.py
LAUNCHER_PATH=dense-9b/recipes/launch_cpt_qwen36_27b_fsdp.sh

# Refuse a SHA that is not a real commit, and refuse a dirty claim about it.
git -C "$REPO" rev-parse --verify "${DEPLOY_SHA}^{commit}" >/dev/null 2>&1 || {
    echo "ERROR: DEPLOY_SHA '$DEPLOY_SHA' is not a commit in $REPO" >&2; exit 1; }

# Hash the OBJECTS, not the working files. This is the value every node must match.
TRAINER_SHA=$(git -C "$REPO" show "$DEPLOY_SHA:$TRAINER_PATH" | sha256sum | awk '{print $1}')
LAUNCHER_SHA=$(git -C "$REPO" show "$DEPLOY_SHA:$LAUNCHER_PATH" | sha256sum | awk '{print $1}')
say "deploying from commit object $DEPLOY_SHA"
say "  trainer  object sha256 $TRAINER_SHA"
say "  launcher object sha256 $LAUNCHER_SHA"

# Corpus is a build artifact, not a commit object, so it carries its own pinned digest.
[ -f "$CORPUS" ] || { echo "ERROR: CORPUS not found: $CORPUS" >&2; exit 1; }
actual_corpus=$(sha256sum "$CORPUS" | awk '{print $1}')
[ "$actual_corpus" = "$CORPUS_SHA256" ] || {
    echo "ERROR: corpus digest mismatch" >&2
    echo "       expected $CORPUS_SHA256" >&2
    echo "       actual   $actual_corpus" >&2
    exit 1; }
say "  corpus   sha256 $actual_corpus (matches pin)"

if [ -n "$ADAPTER" ]; then
    [ -d "$ADAPTER" ] || { echo "ERROR: ADAPTER dir not found: $ADAPTER" >&2; exit 1; }
    _af="$ADAPTER/adapter_model.safetensors"
    [ -f "$_af" ] || { echo "ERROR: no adapter_model.safetensors in $ADAPTER" >&2; exit 1; }
    _actual_adapter=$(sha256sum "$_af" | awk '{print $1}')
    [ "$_actual_adapter" = "$ADAPTER_SHA256" ] || {
        echo "ERROR: adapter digest mismatch — this is the check REQUIRE_LORA_INIT_PARITY cannot make" >&2
        echo "       expected $ADAPTER_SHA256" >&2
        echo "       actual   $_actual_adapter" >&2
        echo "       Parity would have passed four ranks holding this same wrong adapter." >&2
        exit 1; }
    say "  adapter  sha256 $_actual_adapter (matches pin)"
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -C "$REPO" show "$DEPLOY_SHA:$TRAINER_PATH"  > "$tmp/trainer.py"
git -C "$REPO" show "$DEPLOY_SHA:$LAUNCHER_PATH" > "$tmp/launcher.sh"

for n in "${NODES[@]}"; do
    scp -q -o ConnectTimeout=8 "$tmp/trainer.py" \
        "$REMOTE_USER@$n:$REMOTE_ROOT/$TRAINER_PATH"  || { say "ABORT: trainer deploy $n"; exit 1; }
    scp -q -o ConnectTimeout=8 "$tmp/launcher.sh" \
        "$REMOTE_USER@$n:$REMOTE_ROOT/$LAUNCHER_PATH" || { say "ABORT: launcher deploy $n"; exit 1; }
    scp -q -o ConnectTimeout=8 "$CORPUS" \
        "$REMOTE_USER@$n:$REMOTE_ROOT/corpus.jsonl"   || { say "ABORT: corpus deploy $n"; exit 1; }

    if [ -n "$ADAPTER" ]; then
        ssh -o ConnectTimeout=8 "$REMOTE_USER@$n" "mkdir -p '$REMOTE_ROOT/adapter'" || { say "ABORT: $n adapter mkdir"; exit 1; }
        # config and chat template travel WITH the weights — a matching adapter under a
        # different config is a different model, and the parity gate cannot see that either.
        for _f in adapter_model.safetensors adapter_config.json chat_template.jinja tokenizer_config.json; do
            [ -f "$ADAPTER/$_f" ] || continue
            scp -q -o ConnectTimeout=8 "$ADAPTER/$_f" "$REMOTE_USER@$n:$REMOTE_ROOT/adapter/$_f" \
                || { say "ABORT: $n adapter $_f"; exit 1; }
        done
        _ra=$(ssh -o ConnectTimeout=8 "$REMOTE_USER@$n" \
              "sha256sum '$REMOTE_ROOT/adapter/adapter_model.safetensors' | awk '{print \$1}'")
        [ "$_ra" = "$ADAPTER_SHA256" ] || { say "ABORT: $n adapter mismatch — got $_ra"; exit 1; }
    fi

    # Verify what LANDED against the object hashes. Full digests, never prefixes:
    # a 16-char comparison is 64 bits and this gate admits training data.
    read -r t l c < <(ssh -o ConnectTimeout=8 "$REMOTE_USER@$n" \
        "sha256sum '$REMOTE_ROOT/$TRAINER_PATH' '$REMOTE_ROOT/$LAUNCHER_PATH' '$REMOTE_ROOT/corpus.jsonl' \
         | awk '{print \$1}' | xargs")
    [ "$t" = "$TRAINER_SHA" ]  || { say "ABORT: $n trainer  mismatch — got $t";  exit 1; }
    [ "$l" = "$LAUNCHER_SHA" ] || { say "ABORT: $n launcher mismatch — got $l"; exit 1; }
    [ "$c" = "$CORPUS_SHA256" ]|| { say "ABORT: $n corpus   mismatch — got $c"; exit 1; }
    say "  ${n} verified against the commit object${ADAPTER:+ + pinned adapter}"
done

say "=== all four match commit $DEPLOY_SHA — deployed bytes ARE canonical bytes ==="
