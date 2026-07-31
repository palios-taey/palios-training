#!/usr/bin/env python3
"""pack_corpus.py — fixed-length sequence packing for 27B CPT.

WHY (exp: first-step fragmentation OOM): variable-length length-bucketing produces differently-shaped
micro-batches every step → the CUDA caching allocator fragments → Adafactor's first-step workspace
can't find a contiguous block → NV_ERR_NO_MEMORY, even with ~48-102GB free. expandable_segments would
defrag but breaks multi-node RDMA. FIXED-SIZE PACKING gives an IDENTICAL micro-batch shape every step
so the allocator reuses the same freed blocks (no fragmentation) AND wastes zero tokens on padding
(the throughput lever). NO TRUNCATION: documents that cross a block boundary continue in the next
block (nothing is dropped except an optional short final remainder).

Input : jsonl with {"text": ...} per line (raw, unchunked corpus).
Output: jsonl with {"input_ids": [<exactly seq_len ints>]} per line. Loss is on all tokens (CPT).
Docs are joined by EOS so the model still sees document boundaries.

Usage: pack_corpus.py <in.jsonl> <out.jsonl> <tokenizer_dir> [seq_len=2560]
"""
import sys, json
from transformers import AutoTokenizer

def main():
    in_path, out_path, tok_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2560
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id — refusing to pack without a doc separator")

    # stream docs → one long token buffer → emit fixed seq_len blocks as the buffer fills
    buf, n_docs, n_blocks, n_tokens = [], 0, 0, 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ids.append(eos)                      # doc separator (learn boundaries)
            buf.extend(ids)
            n_docs += 1
            n_tokens += len(ids)
            # drain complete blocks (a doc longer than seq_len simply spans multiple blocks — NO truncation)
            while len(buf) >= seq_len:
                block, buf = buf[:seq_len], buf[seq_len:]
                fout.write(json.dumps({"input_ids": block}) + "\n")
                n_blocks += 1
        # final remainder: pad the LAST partial block to seq_len with eos so shape stays uniform
        # (only ~<seq_len tokens of padding across the WHOLE corpus — negligible; keeps every block same shape)
        if buf:
            pad = seq_len - len(buf)
            block = buf + [eos] * pad
            fout.write(json.dumps({"input_ids": block, "pad_tail": pad}) + "\n")
            n_blocks += 1
            print(f"  final block padded with {pad} eos (only partial block in the corpus)", flush=True)

    print(f"PACKED: {n_docs} docs, {n_tokens} tokens → {n_blocks} blocks of {seq_len} "
          f"(waste={100.0*(n_blocks*seq_len - n_tokens)/max(n_blocks*seq_len,1):.2f}%)", flush=True)

if __name__ == "__main__":
    main()
