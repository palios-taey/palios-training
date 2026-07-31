#!/usr/bin/env python3
import subprocess, json, sys
raw = subprocess.run(["redis-cli","-h","127.0.0.1","LRANGE","taey:tutor:inbox","0","-1"],
                     capture_output=True, text=True).stdout
seen = set()
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    p = d.get("purpose") or d.get("platform") or "?"
    if "thermal" not in str(p).lower() and d.get("purpose"):
        continue
    if p in seen:
        continue
    seen.add(p)
    txt = d.get("response_text") or d.get("text") or ""
    bar = "=" * 70
    print("\n%s\n### LANE: %s (platform=%s)\n%s" % (bar, p, d.get("platform","?"), bar))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3500
    print(txt[:n])
