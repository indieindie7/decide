"""refit.py: offline calibration variants from calibraw_<name>_<mode>.json (no GPU).

  py -3.13 refit.py calibraw_llama-gemma_letter.json
Compares one global temperature with one temperature per field (fit on the first half, eval on the second).
"""
import json, sys
import calib

raw = json.load(open(sys.argv[1], encoding="utf-8"))
P, L = raw["probs"], raw["labels"]
fields, half = list(L[0]), len(L) // 2
rows = lambda lo, hi, f=None: [(P[i][g], L[i][g]) for i in range(lo, hi) for g in fields
                               if (f is None or g == f) and sum(P[i][g].values())]
fit, ev = rows(0, half), rows(half, len(L))
T = calib.fit_temp(fit)
out = {"raw": calib.report(ev, 1.0), "global_T": round(T, 3), "global": calib.report(ev, T), "per_field": {}}
pooled = []
for f in fields:
    Tf = calib.fit_temp(rows(0, half, f))
    e = rows(half, len(L), f)
    pooled += [(calib.apply_temp(d, Tf), y) for d, y in e]
    out["per_field"][f] = {"T": round(Tf, 3), "raw": calib.report(e, 1.0), "cal": calib.report(e, Tf)}
out["per_field_pooled"] = calib.report(pooled, 1.0)
print(json.dumps(out, indent=1))
