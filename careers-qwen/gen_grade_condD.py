import json, urllib.request, sys
sys.path.insert(0,os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "careers-qwen"))
import q0_3_semantic_regrade as G
URL="http://PRIVATE_SERVE_HOST_1:8000/v1/chat/completions"; MODEL="condD"
def gen(msgs):
    body=json.dumps({"model":MODEL,"messages":msgs,"chat_template_kwargs":{"enable_thinking":False},"max_tokens":1024,"temperature":0}).encode()
    r=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=180).read())["choices"][0]["message"]["content"]
probes=[json.loads(l) for l in open(os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "foundations/careers/training_data/v2/pairs/k3_values_probes.jsonl"))]
rows=[]; sem=0
for p in probes:
    msgs=[m for m in p["messages"] if m["role"] in ("system","user")]
    gold=next(m["content"] for m in p["messages"] if m["role"]=="assistant")
    ans=gen(msgs); v=G.judge(URL,MODEL,gold,ans)
    ok=v.upper().startswith("PASS"); sem+=ok
    rows.append({"id":p["meta"]["example_id"],"gold":gold,"answer":ans,"verdict":v})
    print(f"  {p['meta']['example_id']}: ans={ans.strip()[:35]!r} -> {v[:40]}",flush=True)
json.dump(rows,open(os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "careers-qwen/q0-3_condD_result.json"),"w"),indent=2)
print(f"=== D (SFT probe) SEMANTIC: {sem}/{len(probes)} ({100*sem//len(probes)}%) ===")
