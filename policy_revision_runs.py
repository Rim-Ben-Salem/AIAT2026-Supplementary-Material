import json, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1])
import arms_race_sim as m
A = m.CALIBRATED["r=0.5"]
CB = dict(policy="withdraw", withdraw_rule="cb")
CFG = {
 "default_cap": dict(policy="cap"),
 "default_budget": dict(policy="budget"),
 "default_withdraw": dict(policy="withdraw"),
 "default_cb_200_200": dict(CB, audit_nh=200, audit_na=200),
 "default_cb_2000_200": dict(CB, audit_nh=2000, audit_na=200),
 "default_cb_2000_200_simul": dict(CB, audit_nh=2000, audit_na=200, audit_delta=0.05/60),
 "anchored_cap": {**A, "policy": "cap"},
 "anchored_budget": {**A, "policy": "budget"},
 "anchored_withdraw": {**A, "policy": "withdraw"},
 "anchored_cb_200_200": {**A, **CB, "audit_nh": 200, "audit_na": 200},
 "anchored_cb_2000_200": {**A, **CB, "audit_nh": 2000, "audit_na": 200},
 "anchored_cb_2000_200_simul": {**A, **CB, "audit_nh": 2000, "audit_na": 200, "audit_delta": 0.05/60},
}
jobs = [(k, s) for k in CFG for s in range(20)]
def f(js):
    k, s = js
    return k, s, m.run(seed=s, **CFG[k])
if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(f, jobs)
    out = {}
    for k, s, h in res:
        out.setdefault(k, {})[s] = h
    json.dump(out, open(sys.argv[2], "w"))
