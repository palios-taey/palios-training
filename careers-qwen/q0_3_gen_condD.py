#!/usr/bin/env python3
"""Generate the 11 k3 relate-probe answers from a served base+D endpoint, enable_thinking:false, FULL length."""
import json, argparse, urllib.request
def gen(url, model, messages, timeout=120):
    body=json.dumps({"model":model,"messages":messages,"chat_template_kwargs":{"enable_thinking":False},
        "max_tokens":1024,"temperature":0}).encode()
    req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=timeout).read())["choices"][0]["message"]["content"]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probes",default=os.path.join(os.environ.get("TREASURER_ROOT", os.path.expanduser("~/treasurer")), "foundations/careers/training_data/v2/pairs/k3_values_probes.jsonl"))
    ap.add_argument("--url",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--out",default=os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "careers-qwen/q0-3_condD_answers.jsonl"))
    a=ap.parse_args()
    probes=[json.loads(l) for l in open(a.probes)]
    rows=[]
    for p in probes:
        msgs=[m for m in p["messages"] if m["role"] in ("system","user")]
        ans=gen(a.url,a.model,msgs)
        gold=next(m["content"] for m in p["messages"] if m["role"]=="assistant")
        rows.append({"id":p["meta"]["example_id"],"gold":gold,"answer":ans,"len":len(ans)})
        print(f"  {p['meta']['example_id']}: {len(ans)} chars")
    json.dump(rows,open(a.out,"w"),indent=2)
    print(f"wrote {len(rows)} answers -> {a.out}")
if __name__=="__main__": main()
