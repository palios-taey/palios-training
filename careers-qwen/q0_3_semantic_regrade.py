#!/usr/bin/env python3
"""q0-3 semantic re-grade: replace the deterministic exact+contain@0.7 grade with an LLM judge.
Finding (infra, verified): the saved answers open with a 'Here's a thinking process:' reasoning
preamble; token-overlap contain counts that as noise -> a correct answer scores low ('0%' = grading
artifact, not capability gap). A semantic judge reads for MEANING through the preamble.

Judge = an OpenAI-compatible endpoint (--judge-url/--judge-model). Grades each (gold, answer) PASS/FAIL
on: does the model answer convey the gold's core factual content, grounded, no contradiction? Reports
which deterministic contain@0.7 verdicts FLIP under the semantic judge (the value-add)."""
import os
import json, re, sys, argparse, urllib.request

def _norm(s): return re.sub(r"\s+"," ", str(s).strip().lower())
def contain_grade(answer, gold):
    a,g=_norm(answer),_norm(gold); gt=set(re.findall(r"[a-z0-9]+",g)); at=set(re.findall(r"[a-z0-9]+",a))
    return (len(gt&at)/max(1,len(gt))) if gt else 0.0

JUDGE_SYS=("You are a strict, fair grader. Given a GOLD reference answer and a MODEL answer, decide if the "
 "MODEL answer conveys the SAME CORE FACTUAL CONTENT as the GOLD — grounded, no contradiction, no invented "
 "facts. The model answer may include reasoning/preamble; judge the ANSWER it lands on, not its wording or "
 "verbosity. Paraphrase that preserves meaning = PASS. Missing the core point, contradicting gold, or "
 "inventing facts = FAIL. Reply EXACTLY one line: 'PASS: <=8-word reason' or 'FAIL: <=8-word reason'.")

def judge(url, model, gold, answer, timeout=60):
    body=json.dumps({"model":model,"messages":[{"role":"system","content":JUDGE_SYS},
        {"role":"user","content":f"GOLD:\n{gold}\n\nMODEL ANSWER:\n{answer}\n\nVerdict:"}],
        "chat_template_kwargs":{"enable_thinking":False},"max_tokens":40,"temperature":0}).encode()
    req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=timeout).read())["choices"][0]["message"]["content"].strip()
    return r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",default=os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "careers-qwen/q0-3_results.json"))
    ap.add_argument("--judge-url",default=os.environ.get("THOR2_URL","http://PRIVATE_SERVE_HOST_2:8000/v1/chat/completions"))
    ap.add_argument("--judge-model",default="phase-s-stage2-v1")
    ap.add_argument("--conditions",default="C,A,B")
    ap.add_argument("--contain-thresh",type=float,default=0.7)
    a=ap.parse_args()
    d=json.load(open(a.results)); rows=d["rows"]; conds=a.conditions.split(",")
    flips=[]; summ={c:{"det_pass":0,"sem_pass":0,"flip_up":0,"flip_down":0} for c in conds}
    for r in rows:
        gold=r["gold"]
        for c in conds:
            ans=r[c]["answer"]; det=contain_grade(ans,gold)>=a.contain_thresh
            v=judge(a.judge_url,a.judge_model,gold,ans); sem=v.upper().startswith("PASS")
            summ[c]["det_pass"]+=det; summ[c]["sem_pass"]+=sem
            if sem and not det: summ[c]["flip_up"]+=1; flips.append((r["id"],c,"contain-FAIL->sem-PASS",v))
            if det and not sem: summ[c]["flip_down"]+=1; flips.append((r["id"],c,"contain-PASS->sem-FAIL",v))
    n=len(rows)
    print(f"=== q0-3 SEMANTIC RE-GRADE (n={n} rows, judge={a.judge_model}) ===")
    for c in conds:
        s=summ[c]; print(f"  {c}: deterministic contain@{a.contain_thresh} pass={s['det_pass']}/{n} ({100*s['det_pass']/n:.0f}%)  ->  SEMANTIC pass={s['sem_pass']}/{n} ({100*s['sem_pass']/n:.0f}%)  [flips: +{s['flip_up']} -{s['flip_down']}]")
    print(f"=== {len(flips)} FLIPS ==="); 
    for id_,c,kind,v in flips[:40]: print(f"  {id_} [{c}] {kind}: {v}")
    json.dump({"summary":summ,"flips":flips},open(os.path.join(os.environ.get("REPO_ROOT", os.path.expanduser("~/palios-training")), "careers-qwen/q0-3_semantic_regrade_results.json"),"w"),indent=2)
if __name__=="__main__": main()
