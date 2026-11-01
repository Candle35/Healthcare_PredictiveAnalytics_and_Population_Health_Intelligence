"""Interface contract tests for the HealthConnect admission-proxy model.

Usage as a library (plug in ANY pipeline):
    run_contract_tests(score_fn, sample_df)
where score_fn(df) -> DataFrame with columns `risk_score` and `status` (index = input index).
Usage as a script (tests the delivered package):  python hc_contract_tests.py
Contract: good rows are scored; bad rows are rejected with a reason while the rest of the batch still scores;
soft problems are scored but visible in `status`; the same input always gives the same output.
"""
import sys, json
import numpy as np, pandas as pd


def run_contract_tests(score_fn, sample):
    base = sample.head(5).copy(); results = []
    CAT_COL = next(c for c in ("Diagnosis", "ProvinceName", "Gender") if c in base.columns)

    def check(tid, name, expected, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"EXCEPTION {type(e).__name__}: {str(e)[:140]}"
        results.append(dict(ID=tid, Scenario=name, Expected=expected, Result="PASS" if ok else "FAIL", Detail=detail))

    ref = score_fn(base)
    rej = lambda o: o.status.astype(str).str.startswith("REJECTED").values
    sane = lambda o: bool(o.risk_score.notna().all() and o.risk_score.between(0, 1).all())

    def c1():
        o = score_fn(base); return (len(o) == 5 and sane(o)), f"{len(o)} rows, scores {o.risk_score.min():.3f}-{o.risk_score.max():.3f}"
    def c2():
        o = score_fn(base.head(1)); return (len(o) == 1 and sane(o)), f"{len(o)} row"
    def c3():
        b = base.copy(); b.iloc[0, b.columns.get_loc("Age")] = np.nan; b.iloc[1, b.columns.get_loc("PriorVisits")] = np.nan; o = score_fn(b)
        return (len(o) == 5 and sane(o)), "numeric NaN handled"
    def c4():
        b = base.copy(); b[CAT_COL] = b[CAT_COL].astype(object); b.iloc[0, b.columns.get_loc(CAT_COL)] = np.nan; o = score_fn(b)
        return (len(o) == 5 and sane(o)), "categorical NaN handled"
    def c5():
        b = base.copy(); b[CAT_COL] = b[CAT_COL].astype(object); b.iloc[0, b.columns.get_loc(CAT_COL)] = "Zanzibar"; o = score_fn(b)
        flagged = "unseen" in str(o.status.iloc[0]).lower() or "UNSEEN" in str(o.status.iloc[0])
        return (len(o) == 5 and sane(o) and flagged), f"row status: {str(o.status.iloc[0])[:80]}"
    def c6():
        try: score_fn(base.drop(columns=["Age"]))
        except Exception as e: return ("Age" in str(e)), f"{type(e).__name__}: {str(e)[:110]}"
        return False, "no error raised for missing column"
    def c7():
        b = base.copy(); b["Surname"] = "x"; o = score_fn(b); return np.allclose(o.risk_score, ref.risk_score), "extra column ignored, scores unchanged"
    def c8():
        o = score_fn(base[base.columns[::-1]]); return np.allclose(o.risk_score, ref.risk_score), "column order does not matter"
    def c9():
        b = base.copy(); b["Age"] = b["Age"].astype(int).astype(str); o = score_fn(b); return np.allclose(o.risk_score, ref.risk_score), "numeric strings coerced"
    def c10():
        b = base.copy(); b["Age"] = b["Age"].astype(object); b.iloc[0, b.columns.get_loc("Age")] = "abc"; o = score_fn(b); r = rej(o)
        return (r[0] and not r[1:].any() and o.risk_score.iloc[1:].notna().all()), "row 1 rejected, rows 2-5 scored"
    def c11():
        b = base.copy(); b.iloc[0, b.columns.get_loc("Age")] = 250; b.iloc[1, b.columns.get_loc("PriorVisits")] = -3; o = score_fn(b); r = rej(o)
        return (r[0] and r[1] and not r[2:].any() and o.risk_score.iloc[2:].notna().all()), "Age=250 and PriorVisits=-3 rejected, rest scored"
    def c12():
        b = base.copy(); b["Gender"] = b["Gender"].str.lower(); b[CAT_COL] = b[CAT_COL].astype(str).str.upper(); o = score_fn(b)
        return np.allclose(o.risk_score, ref.risk_score), "case differences normalised"
    def c13():
        o = score_fn(base.iloc[0:0]); return len(o) == 0, "empty input returns empty output"
    def c14():
        a, b_ = score_fn(base), score_fn(base); return np.allclose(a.risk_score, b_.risk_score), "identical output on repeat call"
    def c15():
        b = base.copy(); b.index = [10, 10, 11, 11, 12]; o = score_fn(b); return (len(o) == 5 and sane(o)), "duplicate index handled"
    for tid, nm, ex, f in [("C1","Valid batch of 5","Scores in [0,1], no NaN",c1), ("C2","Single row","Scored",c2), ("C3","Missing numeric values","Scored (imputed)",c3),
        ("C4","Missing categorical value","Scored, flagged",c4), ("C5","Unseen category ('Zanzibar' in a categorical column)","Scored AND visible in status",c5),
        ("C6","Required column missing","Clear error naming the column",c6), ("C7","Extra unexpected column","Ignored, scores unchanged",c7), ("C8","Columns in different order","Same scores",c8),
        ("C9","Numbers sent as text ('45')","Coerced, same scores",c9), ("C10","Non-numeric value ('abc') in Age","Bad row rejected, rest of batch scored",c10),
        ("C11","Impossible values (Age 250, PriorVisits -3)","Bad rows rejected, rest scored",c11), ("C12","Wrong text case ('female','MALARIA')","Normalised, same scores",c12),
        ("C13","Empty input","Empty output, no crash",c13), ("C14","Repeat call determinism","Identical output",c14), ("C15","Duplicate row index","Scored",c15)]:
        check(tid, nm, ex, f)
    return pd.DataFrame(results)


if __name__ == "__main__":
    from hc_scorer import AdmissionProxyScorer
    sc = AdmissionProxyScorer("."); sample = pd.read_csv("sample_input.csv", keep_default_na=False, na_values=[""])
    res = run_contract_tests(lambda d: sc.score(d), sample)
    print(res[["ID", "Scenario", "Result", "Detail"]].to_string(index=False)); n_fail = int((res.Result == "FAIL").sum())
    print(f"\n{len(res)-n_fail}/{len(res)} contract tests passed"); sys.exit(1 if n_fail else 0)
